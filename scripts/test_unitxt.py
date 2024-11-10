# First Party
from instructlab.eval.unitxt import UnitxtEvaluator, UnitxtFileEvaluator


def test_unitxt():
    print("===> Executing 'test_unitxt'...")
    try:
        model_path = "instructlab/granite-7b-lab"
        unitxt_recipe = "card=cards.wnli,template=templates.classification.multi_class.relation.default,max_train_instances=5,loader_limit=20,num_demos=3,demos_pool_size=10"
        unitxt = UnitxtEvaluator(model_path=model_path, unitxt_recipe=unitxt_recipe)
        overall_score, single_scores = unitxt.run()
        print(f"Overall scores: {overall_score}")
        sample_score = "f1_micro,none"
        assert sample_score in overall_score
        assert overall_score[sample_score] > 0
    except Exception as exc:
        print(f"'test_unitxt_branch' failed: {exc}")
        return False

    return True


def test_unitxt_from_user_file():
    model_path = "instructlab/granite-7b-lab"
    file = "scripts/testdata/unitxt/sample_data.csv"
    task = "qa"
    unitxt = UnitxtFileEvaluator(model_path=model_path, file_path=file, task_type=task)
    overall_score, single_scores = unitxt.run()
    print(f"Overall scores: {overall_score}")
    # test with/without gt_output
    # test different task types
    # test llmaaj
    # test bad file?


if __name__ == "__main__":
    test_unitxt_from_user_file()
