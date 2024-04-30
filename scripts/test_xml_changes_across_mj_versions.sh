#!/bin/sh

MUJOCO_VERSIONS=(
    2.2.0
    2.2.1
    2.2.2
    2.3.0
    2.3.1
    2.3.2
    2.3.3
    2.3.4
    2.3.5
    2.3.6
    2.3.7
)

# Create a new python virtual env and install metaworld
python3.10 -m venv xml_verification_env
source xml_verification_env/bin/activate
python -m pip install -e .

# Loop through mj versions, install it and run the script
for version in ${MUJOCO_VERSIONS[@]}; do
    python -m pip install --upgrade "mujoco==$version"

    echo "------------------------"
    python scripts/xml_changes_verification.py "${1:-}"
    echo "------------------------"
done

deactivate
rm -rf xml_verification_env
