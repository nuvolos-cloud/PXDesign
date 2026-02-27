# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0114,C0301
import os
from pathlib import Path

if "PROTENIX_DATA_ROOT_DIR" in os.environ:
    DATA_ROOT_DIR = os.environ["PROTENIX_DATA_ROOT_DIR"]
else:
    # Fall back to PXDESIGN_ROOT (default /pxdesign) so paths are correct when
    # the package is installed into a site-packages directory.
    _pxdesign_root = Path(os.environ.get("PXDESIGN_ROOT", "/pxdesign"))
    DATA_ROOT_DIR = str(_pxdesign_root / "release_data" / "ccd_cache")

# Use CCD cache created by scripts/gen_ccd_cache.py priority. (without date in filename)
# See: docs/prepare_data.md
CCD_COMPONENTS_FILE_PATH = os.path.join(DATA_ROOT_DIR, "components.cif")
CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
    DATA_ROOT_DIR, "components.cif.rdkit_mol.pkl"
)
PDB_CLUSTER_FILE_PATH = os.path.join(DATA_ROOT_DIR, "clusters-by-entity-40.txt")

if (not os.path.exists(CCD_COMPONENTS_FILE_PATH)) or (not os.path.exists(CCD_COMPONENTS_RDKIT_MOL_FILE_PATH)):
    CCD_COMPONENTS_FILE_PATH = os.path.join(
        DATA_ROOT_DIR, "components.v20240608.cif"
    )
    CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
        DATA_ROOT_DIR, "components.v20240608.cif.rdkit_mol.pkl"
    )


data_configs = {
    "ccd_components_file": CCD_COMPONENTS_FILE_PATH,
    "ccd_components_rdkit_mol_file": CCD_COMPONENTS_RDKIT_MOL_FILE_PATH,
    "pdb_cluster_file": PDB_CLUSTER_FILE_PATH,
}
