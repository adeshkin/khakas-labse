from huggingface_hub import HfApi

def main():
    # https://huggingface.co/settings/tokens
    api = HfApi(token="") # todo add token

    api.create_repo(
        repo_id="adeshkin/labse-kjh-rus1",
        repo_type='model',
        exist_ok=True
    )

    api.upload_folder(
        repo_id="adeshkin/labse-kjh-rus1",
        folder_path="./output1/best_model_kjh_rus",
        repo_type="model",
    )