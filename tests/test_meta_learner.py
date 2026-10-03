from sizon.research.meta_learner import MetaLearner

def test_meta_learner_fit_predict(sample_feed, example_genome):
    learner = MetaLearner(n_estimators=5, random_state=42)
    
    # Fit the meta-learner
    learner.fit(sample_feed, [example_genome])
    
    # Predict
    preds = learner.predict(sample_feed)
    
    assert len(preds) == len(sample_feed.rows)
    assert all(isinstance(p, float) for p in preds)
