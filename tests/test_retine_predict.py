import unittest
import numpy as np
from modules.heart_failure.predict import load_artifact, predict, to_features

class RetineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.art = load_artifact()

    def inputs(self):
        row = dict(self.art["defaults"])
        row.update(ma1=round(row["ma1"]), ma5=round(row["ma5"]), quality=1)
        return row

    def test_centres_preserve_labels(self):
        for label, centre in enumerate(self.art["model"].cluster_centers_):
            raw = self.art["scaler"].inverse_transform([centre])[0]
            row = dict(zip(self.art["features"], raw))
            row.update(quality=1, ma1=round(row["ma1"]), ma5=round(row["ma5"]))
            self.assertEqual(predict(row, self.art)["cluster"], label)

    def test_invalid_inputs(self):
        for edits in ({"quality":0}, {"ma1":1.2}, {"exudate1":float("nan")},
                      {"optic_disc_diameter":0}, {"exudate2":-1}, {"ma5":None}):
            with self.subTest(edits=edits):
                with self.assertRaises(ValueError):
                    predict({**self.inputs(), **edits}, self.art)

    def test_column_order_and_no_refit(self):
        row=self.inputs()
        mean=self.art["scaler"].mean_.copy()
        centres=self.art["model"].cluster_centers_.copy()
        self.assertEqual(predict(row,self.art)["cluster"],predict(dict(reversed(list(row.items()))),self.art)["cluster"])
        self.assertEqual(to_features(row,self.art).columns.tolist(),self.art["features"])
        np.testing.assert_array_equal(mean,self.art["scaler"].mean_)
        np.testing.assert_array_equal(centres,self.art["model"].cluster_centers_)

if __name__ == "__main__":
    unittest.main()
