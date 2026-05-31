import unittest

from config.settings import EngineConfig, EnvConfig, ModelConfig


class TestConfigValidation(unittest.TestCase):
    def test_valid_defaults(self):
        EngineConfig()
        EnvConfig()
        ModelConfig()

    def test_invalid_engine_thresholds_raise(self):
        with self.assertRaises(ValueError):
            EngineConfig(eps_reg=-0.1)
        with self.assertRaises(ValueError):
            EngineConfig(eps_var=-0.1)
        with self.assertRaises(ValueError):
            EngineConfig(num_samples=0)
        with self.assertRaises(ValueError):
            EngineConfig(confidence_percentile=101.0)

    def test_invalid_environment_config_raises(self):
        with self.assertRaises(ValueError):
            EnvConfig(alpha=1.1)
        with self.assertRaises(ValueError):
            EnvConfig(price_fluctuation=-0.1)
        with self.assertRaises(ValueError):
            EnvConfig(data_path="")

    def test_invalid_model_config_raises(self):
        with self.assertRaises(ValueError):
            ModelConfig(sigma2=0.0)


if __name__ == "__main__":
    unittest.main()
