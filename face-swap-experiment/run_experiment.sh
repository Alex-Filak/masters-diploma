#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
FACEFUSION_DIR="${ROOT_DIR}/facefusion"
INPUT_DIR="${ROOT_DIR}/test-img"
EXPERIMENT_DIR="${SCRIPT_DIR}"
OUTPUT_ROOT="${EXPERIMENT_DIR}/processed_data"
LOG_ROOT="${EXPERIMENT_DIR}/logs"

FEMALE_REF="${INPUT_DIR}/female_prot.png"
MALE_REF="${INPUT_DIR}/male_prot.png"

if [[ -z "${DIPLOMA_ENV_READY:-}" ]]; then
	exec nix develop "${ROOT_DIR}" -c bash "${BASH_SOURCE[0]}" "$@"
fi

cd "${FACEFUSION_DIR}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

require_file() {
	local path="$1"

	if [[ ! -f "${path}" ]]; then
		echo "Missing required file: ${path}" >&2
		exit 1
	fi
}

run_facefusion() {
	local model="$1"
	local preset="$2"
	local source_path="$3"
	local target_path="$4"
	local output_path="$5"
	shift 5

	local output_dir
	local log_path
	output_dir="$(dirname "${output_path}")"
	log_path="${LOG_ROOT}/${model}/${preset}/$(basename "${output_path}" .png).log"

	mkdir -p "${output_dir}" "$(dirname "${log_path}")"

	if [[ -s "${output_path}" ]]; then
		echo "Skipping ${model}/${preset}: $(basename "${target_path}")"
		return
	fi

	echo "Processing ${model}/${preset}: $(basename "${target_path}")"

	python facefusion.py headless-run \
		--source-paths "${source_path}" \
		--target-path "${target_path}" \
		--output-path "${output_path}" \
		--execution-providers cuda \
		--execution-device-ids 0 \
		--output-image-quality 100 \
		--face-selector-mode one \
		--face-swapper-model "${model}" \
		"$@" \
		--log-level info \
		>"${log_path}" 2>&1
}

preset_args() {
	local preset="$1"

	case "${preset}" in
		A)
			printf '%s\n' \
				--face-swapper-pixel-boost 512x512 \
				--face-swapper-weight 0.8 \
				--face-mask-types box occlusion \
				--face-mask-blur 0.3 \
				--face-mask-padding 0 0 0 0 \
				--processors face_swapper
			;;
		B)
			printf '%s\n' \
				--face-swapper-pixel-boost 512x512 \
				--face-swapper-weight 1.0 \
				--face-mask-types box occlusion \
				--face-mask-blur 0.3 \
				--face-mask-padding 10 10 10 10 \
				--processors face_swapper
			;;
		C)
			printf '%s\n' \
				--face-swapper-pixel-boost 512x512 \
				--face-swapper-weight 0.9 \
				--face-mask-types box occlusion \
				--face-mask-blur 0.5 \
				--face-mask-padding 15 15 15 15 \
				--processors face_swapper
			;;
		D)
			printf '%s\n' \
				--face-swapper-pixel-boost 768x768 \
				--face-swapper-weight 1.0 \
				--face-mask-types box occlusion \
				--face-mask-blur 0.3 \
				--face-mask-padding 10 10 10 10 \
				--processors face_swapper
			;;
		E)
			printf '%s\n' \
				--face-swapper-pixel-boost 512x512 \
				--face-swapper-weight 0.9 \
				--face-mask-types box occlusion \
				--face-mask-blur 0.3 \
				--face-mask-padding 10 10 10 10 \
				--processors face_swapper face_enhancer
			;;
		*)
			echo "Unknown preset: ${preset}" >&2
			exit 1
			;;
	esac
}

run_model_preset() {
	local model="$1"
	local preset="$2"
	local preset_label="preset_${preset}"
	local args=()

	mapfile -t args < <(preset_args "${preset}")

	if [[ "${model}" == "simswap_unofficial_512" && "${preset}" == "D" ]]; then
		preset_label="preset_D_1024x1024"
		for index in "${!args[@]}"; do
			if [[ "${args[${index}]}" == "768x768" ]]; then
				args[${index}]="1024x1024"
			fi
		done
	fi

	for index in 1 2 3 4 5; do
		run_facefusion \
			"${model}" \
			"${preset_label}" \
			"${FEMALE_REF}" \
			"${INPUT_DIR}/female${index}.png" \
			"${OUTPUT_ROOT}/${model}/${preset_label}/female/female${index}.png" \
			"${args[@]}"

		run_facefusion \
			"${model}" \
			"${preset_label}" \
			"${MALE_REF}" \
			"${INPUT_DIR}/male${index}.png" \
			"${OUTPUT_ROOT}/${model}/${preset_label}/male/male${index}.png" \
			"${args[@]}"
	done
}

require_file "${FEMALE_REF}"
require_file "${MALE_REF}"
for index in 1 2 3 4 5; do
	require_file "${INPUT_DIR}/female${index}.png"
	require_file "${INPUT_DIR}/male${index}.png"
done

run_model_preset hyperswap_1a_256 A
run_model_preset hyperswap_1a_256 B
run_model_preset hyperswap_1a_256 C
run_model_preset hyperswap_1a_256 D
run_model_preset hyperswap_1a_256 E

run_model_preset hyperswap_1c_256 A
run_model_preset hyperswap_1c_256 B
run_model_preset hyperswap_1c_256 D

run_model_preset inswapper_128_fp16 A
run_model_preset inswapper_128_fp16 B
run_model_preset inswapper_128_fp16 C
run_model_preset inswapper_128_fp16 E

run_model_preset simswap_unofficial_512 A
run_model_preset simswap_unofficial_512 B
run_model_preset simswap_unofficial_512 D

echo "Experiment finished. Results: ${OUTPUT_ROOT}"
