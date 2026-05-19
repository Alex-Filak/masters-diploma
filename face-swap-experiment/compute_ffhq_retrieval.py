#!/usr/bin/env python
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy
import pandas
import torch

from compute_face_metrics import (
	INPUT_DIR,
	MATRIX,
	OUTPUT_ROOT,
	REPORTS_DIR,
	clear_static_faces,
	get_many_faces,
	get_one_face,
	l2_normalize,
	load_adaface,
	setup_facefusion,
	state_manager,
	warp_face_by_face_landmark_5
)

FFHQ_ARCHIVE_PATH = ROOT_DIR / 'ffhq.tar.gz'
GALLERY_INDEX_PATH = REPORTS_DIR / 'ffhq_gallery_embeddings.npz'
GALLERY_MANIFEST_PATH = REPORTS_DIR / 'ffhq_gallery_manifest.csv'
RETRIEVAL_RESULTS_PATH = REPORTS_DIR / 'ffhq_retrieval_results.csv'
RETRIEVAL_QUERY_SUMMARY_PATH = REPORTS_DIR / 'ffhq_retrieval_query_summary.csv'
RETRIEVAL_EXPERIMENT_SUMMARY_PATH = REPORTS_DIR / 'ffhq_retrieval_experiment_summary.csv'
GALLERY_ASSET_DIR = REPORTS_DIR / 'notebook_assets' / 'ffhq_gallery'


def decode_member_image(archive_file: tarfile.ExFileObject | None) -> numpy.ndarray | None:
	if archive_file is None:
		return None
	data = numpy.frombuffer(archive_file.read(), dtype=numpy.uint8)
	if data.size == 0:
		return None
	return cv2.imdecode(data, cv2.IMREAD_COLOR)


def image_embeddings_from_frame(frame: numpy.ndarray, adaface_model) -> Dict[str, numpy.ndarray]:
	faces = get_many_faces([ frame ])
	face = get_one_face(faces, 0)
	if face is None:
		raise ValueError('No face detected')
	return {
		'face': face,
		'arcface': l2_normalize(face.embedding_norm),
		'adaface': adaface_embedding_on_device(adaface_model, frame, face)
	}


def adaface_embedding_on_device(model: torch.nn.Module, frame: numpy.ndarray, face) -> numpy.ndarray:
	crop, _ = warp_face_by_face_landmark_5(frame, face.landmark_set.get('5/68'), 'arcface_112_v2', (112, 112))
	crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
	crop = crop.astype(numpy.float32) / 127.5 - 1.0
	device = next(model.parameters()).device
	tensor = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0).to(device)
	with torch.no_grad():
		embedding = model(tensor)
	if isinstance(embedding, (tuple, list)):
		embedding = embedding[0]
	return l2_normalize(embedding.detach().cpu().numpy())


def configure_execution() -> torch.nn.Module:
	setup_facefusion()
	state_manager.set_item('execution_providers', [ 'cuda' ])
	state_manager.set_item('execution_device_ids', [ 0 ])
	adaface_model = load_adaface()
	if torch.cuda.is_available():
		adaface_model = adaface_model.to('cuda')
		print('AdaFace device: cuda')
	else:
		print('AdaFace device: cpu')
	print(f"FaceFusion execution providers: {state_manager.get_item('execution_providers')}")
	return adaface_model


def build_gallery_index(adaface_model) -> Tuple[pandas.DataFrame, numpy.ndarray, numpy.ndarray]:
	if GALLERY_INDEX_PATH.is_file() and GALLERY_MANIFEST_PATH.is_file():
		manifest = pandas.read_csv(GALLERY_MANIFEST_PATH)
		index = numpy.load(GALLERY_INDEX_PATH, allow_pickle=True)
		return manifest, index['arcface'], index['adaface']

	rows: List[Dict[str, object]] = []
	arcface_embeddings: List[numpy.ndarray] = []
	adaface_embeddings: List[numpy.ndarray] = []
	processed = 0

	with tarfile.open(FFHQ_ARCHIVE_PATH, 'r:gz') as archive:
		for member in archive:
			if not member.isfile() or not member.name.startswith('ffhq/Part1/') or not member.name.endswith('.png'):
				continue
			frame = decode_member_image(archive.extractfile(member))
			if frame is None:
				continue
			try:
				embeddings = image_embeddings_from_frame(frame, adaface_model)
			except ValueError:
				clear_static_faces()
				continue
			face = embeddings['face']
			clear_static_faces()
			if face.angle != 0:
				continue

			rows.append({
				'gallery_member_path': member.name,
				'gallery_asset_path': str((GALLERY_ASSET_DIR / Path(member.name).name).relative_to(REPORTS_DIR.parent)),
				'face_angle': int(face.angle),
				'gender': face.gender
			})
			arcface_embeddings.append(embeddings['arcface'])
			adaface_embeddings.append(embeddings['adaface'])
			processed += 1
			if processed % 250 == 0:
				print(f'Indexed {processed} FFHQ frontal faces')

	manifest = pandas.DataFrame(rows)
	arcface_array = numpy.vstack(arcface_embeddings)
	adaface_array = numpy.vstack(adaface_embeddings)
	manifest.to_csv(GALLERY_MANIFEST_PATH, index=False)
	numpy.savez_compressed(
		GALLERY_INDEX_PATH,
		arcface=arcface_array,
		adaface=adaface_array
	)
	print(f'Built FFHQ gallery index with {len(manifest)} frontal faces')
	return manifest, arcface_array, adaface_array


def result_paths():
	for model, presets in MATRIX.items():
		for preset in presets:
			for gender in [ 'female', 'male' ]:
				for index in range(1, 6):
					image_name = f'{gender}{index}.png'
					yield {
						'model': model,
						'preset': preset,
						'gender': gender,
						'image_index': index,
						'image_name': image_name,
						'original_path': INPUT_DIR / image_name,
						'modified_path': OUTPUT_ROOT / model / preset / gender / image_name
					}


def query_embeddings(adaface_model) -> Tuple[Dict[Path, Dict[str, numpy.ndarray]], Dict[str, Dict[str, object]]]:
	cache: Dict[Path, Dict[str, numpy.ndarray]] = {}
	original_query_map: Dict[str, Dict[str, object]] = {}
	for query in result_paths():
		original_path = query['original_path']
		modified_path = query['modified_path']
		for path in [ original_path, modified_path ]:
			if path in cache:
				continue
			frame = cv2.imread(str(path))
			if frame is None:
				raise ValueError(f'Cannot read query image: {path}')
			embeddings = image_embeddings_from_frame(frame, adaface_model)
			clear_static_faces()
			cache[path] = {
				'arcface': embeddings['arcface'],
				'adaface': embeddings['adaface']
			}
		original_query_map[query['image_name']] = {
			'gender': query['gender'],
			'image_index': query['image_index'],
			'image_name': query['image_name'],
			'original_path': original_path
		}
	return cache, original_query_map


def extract_gallery_assets(member_paths: List[str]) -> None:
	GALLERY_ASSET_DIR.mkdir(parents=True, exist_ok=True)
	required = { member_path: GALLERY_ASSET_DIR / Path(member_path).name for member_path in member_paths }
	missing = { member_path for member_path, asset_path in required.items() if not asset_path.is_file() }
	if not missing:
		return

	with tarfile.open(FFHQ_ARCHIVE_PATH, 'r:gz') as archive:
		for member in archive:
			if member.name not in missing:
				continue
			frame = decode_member_image(archive.extractfile(member))
			if frame is None:
				continue
			cv2.imwrite(str(required[member.name]), frame)


def top_k(similarities: numpy.ndarray, limit: int = 5) -> numpy.ndarray:
	limit = min(limit, similarities.shape[0])
	indices = numpy.argpartition(-similarities, limit - 1)[:limit]
	return indices[numpy.argsort(-similarities[indices])]


def main() -> None:
	REPORTS_DIR.mkdir(parents=True, exist_ok=True)
	adaface_model = configure_execution()
	gallery_manifest, gallery_arcface, gallery_adaface = build_gallery_index(adaface_model)
	query_cache, original_query_map = query_embeddings(adaface_model)

	gallery_embeddings = {
		'arcface': gallery_arcface,
		'adaface': gallery_adaface
	}
	source_gallery_match: Dict[Tuple[str, str], str] = {}

	for image_name, query_info in original_query_map.items():
		original_path = query_info['original_path']
		for recognizer in [ 'arcface', 'adaface' ]:
			similarities = gallery_embeddings[recognizer] @ query_cache[original_path][recognizer]
			top_indices = top_k(similarities, 1)
			source_gallery_match[(image_name, recognizer)] = str(gallery_manifest.iloc[top_indices[0]]['gallery_member_path'])

	result_rows = []
	query_rows = []

	for query in result_paths():
		source_path = query['original_path']
		modified_path = query['modified_path']
		for recognizer in [ 'arcface', 'adaface' ]:
			query_embedding = query_cache[modified_path][recognizer]
			similarities = gallery_embeddings[recognizer] @ query_embedding
			top_indices = top_k(similarities, 5)
			source_match_member = source_gallery_match[(query['image_name'], recognizer)]

			source_match_top1 = False
			source_match_top5 = False
			source_gallery_distance = None
			source_row = gallery_manifest.index[gallery_manifest['gallery_member_path'] == source_match_member]
			if not source_row.empty:
				source_gallery_index = int(source_row[0])
				source_gallery_distance = float(1.0 - similarities[source_gallery_index])

			for rank, gallery_index in enumerate(top_indices, start=1):
				gallery_info = gallery_manifest.iloc[int(gallery_index)]
				gallery_member = str(gallery_info['gallery_member_path'])
				is_source_gallery_match = gallery_member == source_match_member
				source_match_top1 = source_match_top1 or (rank == 1 and is_source_gallery_match)
				source_match_top5 = source_match_top5 or is_source_gallery_match
				result_rows.append({
					'model': query['model'],
					'preset': query['preset'],
					'gender': query['gender'],
					'image_index': query['image_index'],
					'image_name': query['image_name'],
					'original_path': str(source_path),
					'modified_path': str(modified_path),
					'ranker_recognizer': recognizer,
					'rank': rank,
					'gallery_member_path': gallery_member,
					'gallery_asset_path': str(gallery_info['gallery_asset_path']),
					'ranker_cos_distance': float(1.0 - similarities[int(gallery_index)]),
					'gallery_cos_distance_arcface': float(1.0 - numpy.dot(gallery_arcface[int(gallery_index)], query_cache[modified_path]['arcface'])),
					'gallery_cos_distance_adaface': float(1.0 - numpy.dot(gallery_adaface[int(gallery_index)], query_cache[modified_path]['adaface'])),
					'is_source_gallery_match': is_source_gallery_match
				})

			query_rows.append({
				'model': query['model'],
				'preset': query['preset'],
				'gender': query['gender'],
				'image_index': query['image_index'],
				'image_name': query['image_name'],
				'ranker_recognizer': recognizer,
				'source_gallery_member_path': source_match_member,
				'source_gallery_distance': source_gallery_distance,
				'source_match_top1': source_match_top1,
				'source_match_top5': source_match_top5,
				'top1_ranker_distance': float(1.0 - similarities[int(top_indices[0])]),
				'top1_gallery_member_path': str(gallery_manifest.iloc[int(top_indices[0])]['gallery_member_path'])
			})

	results = pandas.DataFrame(result_rows)
	query_summary = pandas.DataFrame(query_rows)
	experiment_summary = (
		query_summary
		.groupby([ 'model', 'preset', 'ranker_recognizer' ], as_index=False)
		.agg(
			mean_top1_ranker_distance=('top1_ranker_distance', 'mean'),
			mean_source_gallery_distance=('source_gallery_distance', 'mean'),
			source_match_top1_rate=('source_match_top1', 'mean'),
			source_match_top5_rate=('source_match_top5', 'mean'),
			image_count=('image_name', 'count')
		)
		.sort_values([ 'ranker_recognizer', 'source_match_top5_rate', 'mean_source_gallery_distance' ], ascending=[ True, True, False ])
	)

	results.to_csv(RETRIEVAL_RESULTS_PATH, index=False)
	query_summary.to_csv(RETRIEVAL_QUERY_SUMMARY_PATH, index=False)
	experiment_summary.to_csv(RETRIEVAL_EXPERIMENT_SUMMARY_PATH, index=False)
	extract_gallery_assets(results['gallery_member_path'].drop_duplicates().tolist())

	print(f'Wrote {len(results)} FFHQ retrieval rows')
	print(f'Wrote {len(query_summary)} query summary rows')
	print('Best source-suppression setup per recognizer:')
	print(experiment_summary.groupby('ranker_recognizer').head(1).to_string(index=False))


if __name__ == '__main__':
	main()
