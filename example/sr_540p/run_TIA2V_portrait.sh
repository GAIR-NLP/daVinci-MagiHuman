#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BR_WIDTH="${BR_WIDTH:-256}"
export BR_HEIGHT="${BR_HEIGHT:-448}"
export SR_WIDTH="${SR_WIDTH:-512}"
export SR_HEIGHT="${SR_HEIGHT:-896}"

exec bash "${SCRIPT_DIR}/run_TIA2V.sh"
