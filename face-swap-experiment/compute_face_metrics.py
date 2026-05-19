#!/usr/bin/env python
from __future__ import annotations

import contextlib
import io
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy
import pandas
import torch
from transformers import AutoModel

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parent
FACEFUSION_DIR = ROOT_DIR / 'facefusion'
INPUT_DIR = ROOT_DIR / 'test-img'
OUTPUT_ROOT = EXPERIMENT_DIR / 'processed_data'
REPORTS_DIR = EXPERIMENT_DIR / 'reports'

ADAFACE_MODEL_ID = 'minchul/cvlface_adaface_ir50_ms1mv2'

sys.path.insert(0, str(FACEFUSION_DIR))

from facefusion import face_classifier, face_detector, face_landmarker, face_recognizer, state_manager  # noqa: E402
from facefusion.face_analyser import get_many_faces, get_one_face  # noqa: E402
from facefusion.face_helper import warp_face_by_face_landmark_5  # noqa: E402
from facefusion.face_store import clear_static_faces  # noqa: E402
from facefusion.vision import read_static_image  # noqa: E402


MATRIX: Dict[str, List[str]] = {
	'hyperswap_1a_256': [ 'preset_A', 'preset_B', 'preset_C', 'preset_D', 'preset_E' ],
	'hyperswap_1c_256': [ 'preset_A', 'preset_B', 'preset_D' ],
	'inswapper_128_fp16': [ 'preset_A', 'preset_B', 'preset_C', 'preset_E' ],
	'simswap_unofficial_512': [ 'preset_A', 'preset_B', 'preset_D_1024x1024' ]
}

PRESET_LABELS: Dict[str, str] = {
	'preset_A': 'A - baseline',
	'preset_B': 'B - strong identity replacement',
	'preset_C': 'C - smoother boundary',
	'preset_D': 'D - high detail',
	'preset_E': 'E - enhanced output',
	'preset_D_1024x1024': 'D - high detail, 1024x1024 SimSwap workaround'
}


def setup_facefusion() -> None:
	state_manager.init_item('execution_device_ids', [ 0 ])
	state_manager.init_item('execution_providers', [ 'cpu' ])
	state_manager.init_item('download_providers', [ 'github', 'huggingface' ])
	state_manager.init_item('face_detector_angles', [ 0 ])
	state_manager.init_item('face_detector_model', 'yolo_face')
	state_manager.init_item('face_detector_size', '640x640')
	state_manager.init_item('face_detector_margin', (0, 0, 0, 0))
	state_manager.init_item('face_detector_score', 0.5)
	state_manager.init_item('face_landmarker_model', '2dfan4')
	state_manager.init_item('face_landmarker_score', 0.5)
	face_detector.pre_check()
	face_landmarker.pre_check()
	face_classifier.pre_check()
	face_recognizer.pre_check()


def load_adaface() -> torch.nn.Module:
	from huggingface_hub import snapshot_download

	snapshot_path = Path(snapshot_download(ADAFACE_MODEL_ID))
	sys.path.insert(0, str(snapshot_path))

	previous_cwd = Path.cwd()
	os.chdir(snapshot_path)
	try:
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
				model = AutoModel.from_pretrained(str(snapshot_path), trust_remote_code=True)
	finally:
		os.chdir(previous_cwd)

	model.eval()
	return model


def l2_normalize(embedding: numpy.ndarray) -> numpy.ndarray:
	embedding = embedding.astype(numpy.float32).reshape(-1)
	norm = numpy.linalg.norm(embedding)
	if norm == 0:
		raise ValueError('Zero embedding encountered')
	return embedding / norm


def cosine_distance(a: numpy.ndarray, b: numpy.ndarray) -> float:
	return float(1.0 - numpy.dot(a, b))


def read_face(path: Path):
	frame = read_static_image(str(path))
	if frame is None:
		raise ValueError(f'Cannot read image: {path}')

	faces = get_many_faces([ frame ])
	face = get_one_face(faces, 0)
	if face is None:
		raise ValueError(f'No face detected: {path}')
	return frame, face


def adaface_embedding(model: torch.nn.Module, frame: numpy.ndarray, face) -> numpy.ndarray:
	crop, _ = warp_face_by_face_landmark_5(frame, face.landmark_set.get('5/68'), 'arcface_112_v2', (112, 112))
	crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
	crop = crop.astype(numpy.float32) / 127.5 - 1.0
	tensor = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0)
	with torch.no_grad():
		embedding = model(tensor)
	if isinstance(embedding, (tuple, list)):
		embedding = embedding[0]
	return l2_normalize(embedding.detach().cpu().numpy())


def image_embeddings(path: Path, adaface_model: torch.nn.Module) -> Dict[str, numpy.ndarray]:
	frame, face = read_face(path)
	return {
		'arcface': l2_normalize(face.embedding_norm),
		'adaface': adaface_embedding(adaface_model, frame, face)
	}


def result_paths() -> Iterable[Tuple[str, str, str, int, Path, Path, Path]]:
	for model, presets in MATRIX.items():
		for preset in presets:
			for gender in [ 'female', 'male' ]:
				reference_path = INPUT_DIR / f'{gender}_prot.png'
				for index in range(1, 6):
					image_name = f'{gender}{index}.png'
					original_path = INPUT_DIR / image_name
					modified_path = OUTPUT_ROOT / model / preset / gender / image_name
					yield model, preset, gender, index, original_path, reference_path, modified_path


def main() -> None:
	REPORTS_DIR.mkdir(parents=True, exist_ok=True)
	setup_facefusion()
	adaface_model = load_adaface()
	embedding_cache: Dict[Path, Dict[str, numpy.ndarray]] = {}
	rows = []

	for model, preset, gender, index, original_path, reference_path, modified_path in result_paths():
		for path in [ original_path, reference_path, modified_path ]:
			if path not in embedding_cache:
				embedding_cache[path] = image_embeddings(path, adaface_model)

		for recognizer in [ 'arcface', 'adaface' ]:
			original_embedding = embedding_cache[original_path][recognizer]
			reference_embedding = embedding_cache[reference_path][recognizer]
			modified_embedding = embedding_cache[modified_path][recognizer]
			original_distance = cosine_distance(original_embedding, modified_embedding)
			reference_distance = cosine_distance(reference_embedding, modified_embedding)
			rows.append({
				'model': model,
				'preset': preset,
				'preset_label': PRESET_LABELS[preset],
				'gender': gender,
				'image_index': index,
				'image_name': f'{gender}{index}.png',
				'original_path': str(original_path),
				'reference_path': str(reference_path),
				'modified_path': str(modified_path),
				'recognizer': recognizer,
				'cos_distance_original_modified': original_distance,
				'cos_distance_reference_modified': reference_distance,
				'cos_similarity_original_modified': 1.0 - original_distance,
				'cos_similarity_reference_modified': 1.0 - reference_distance,
				'identity_replacement_score': original_distance - reference_distance
			})
		clear_static_faces()

	results = pandas.DataFrame(rows)
	results.to_csv(REPORTS_DIR / 'face_similarity_results.csv', index=False)

	summary = (
		results
		.groupby([ 'model', 'preset', 'preset_label', 'recognizer' ], as_index=False)
		.agg(
			mean_original_distance=('cos_distance_original_modified', 'mean'),
			mean_reference_distance=('cos_distance_reference_modified', 'mean'),
			mean_original_similarity=('cos_similarity_original_modified', 'mean'),
			mean_reference_similarity=('cos_similarity_reference_modified', 'mean'),
			mean_identity_replacement_score=('identity_replacement_score', 'mean'),
			min_original_distance=('cos_distance_original_modified', 'min'),
			max_reference_distance=('cos_distance_reference_modified', 'max'),
			image_count=('image_name', 'count')
		)
		.sort_values([ 'recognizer', 'mean_identity_replacement_score' ], ascending=[ True, False ])
	)
	summary.to_csv(REPORTS_DIR / 'face_similarity_summary.csv', index=False)

	combined = (
		summary
		.groupby([ 'model', 'preset', 'preset_label' ], as_index=False)
		.agg(
			mean_original_distance=('mean_original_distance', 'mean'),
			mean_reference_distance=('mean_reference_distance', 'mean'),
			mean_identity_replacement_score=('mean_identity_replacement_score', 'mean'),
			arcface_score=('mean_identity_replacement_score', lambda values: float(values.iloc[0])),
			image_count=('image_count', 'min')
		)
		.sort_values('mean_identity_replacement_score', ascending=False)
	)
	combined.to_csv(REPORTS_DIR / 'face_similarity_combined_summary.csv', index=False)

	print(f'Wrote {len(results)} measurement rows')
	print(f'Wrote {len(summary)} recognizer summary rows')
	print('Best combined setup:')
	print(combined.head(1).to_string(index=False))


if __name__ == '__main__':
	main()
