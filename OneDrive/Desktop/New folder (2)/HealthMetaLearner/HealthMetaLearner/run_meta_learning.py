# run_meta_learning.py
from health_meta_learner.meta_learning import HierarchicalMetaLearner

if __name__ == "__main__":
    # Instantiate the HierarchicalMetaLearner
    meta_learner = HierarchicalMetaLearner()

    # Simulated healthcare data (replace this with actual data loading)
    healthcare_data = {
        "EHRs": np.random.randn(100, 50),        # Example: Electronic Health Records data
        "Imaging": np.random.randn(100, 30),     # Example: Radiological Imaging data
        "GenomicData": np.random.randn(100, 20), # Example: Genomic data
        "PROs": np.random.randn(100, 10),        # Example: Patient-Reported Outcomes data
        "BehavioralData": np.random.randn(100, 15),  # Example: Behavioral data
    }

    # Perform hierarchical meta-learning
    trained_model = meta_learner.meta_learn(healthcare_data)

    print("Hierarchical Meta-Learning completed.")
