"""Vérifie que la plateforme redonne les mêmes prédictions que le notebook (20 exemples sauvegardés)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from modules.bone_marrow import predict as P


def main():
    art = P.load_artifact()
    expected = np.clip(art["sample_pred"], art["target"]["lo"], art["target"]["hi"])
    for i, row in art["sample_inputs"].iterrows():
        got = P.predict(row.to_dict(), art, check=False)["days"]
        assert abs(got - expected[i]) < 1e-6, f"ligne {i} : plateforme {got} vs notebook {expected[i]}"
    print("OK : 20 prédictions identiques au notebook")


if __name__ == "__main__":
    main()