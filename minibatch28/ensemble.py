import os
import pandas as pd
import numpy as np
import glob
import argparse


def load_data_list(data_dir: str) -> list:
    """
    ensemble 할 대상이 들어있는 directory 내의 csv file들의 이름이 담긴 list로 반환합니다.
    """
    csv_list = glob.glob(f"{data_dir}/*.csv")

    return csv_list


def sum_and_get_argmax(df_list: list) -> pd.DataFrame:
    """
    DataFrame들을 모두 합하고 argmax한 값들을 ans column으로 추가합니다.
    ans column을 추가한 DataFrame을 반환합니다.
    """
    result = sum(df_list)
    result["ans"] = result.apply(lambda x: np.argmax(x), axis=1)

    return result


def get_submission_df(df: pd.DataFrame, col: pd.Series) -> pd.DataFrame:
    """
    최종 ans를 구한 후, ImageID를 다시 붙이고 ans만 남겨서
    submission DataFame을 만듭니다.
    col은 ImageID 값이 담긴 Series입니다.
    """
    df = pd.concat([col, df], axis=1)

    return df[["ImageID", "ans"]]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # ensemble.py --data_dir "ouput csv파일들이 들어있는 폴더 경로"
    parser.add_argument('--output_dir', type=str, default="./output")
    args = parser.parse_args()

    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    data_list = load_data_list(output_dir)

    # csv file을 읽어와 dataframe으로 반환합니다.
    df_list = [pd.read_csv(file_name) for file_name in data_list]

    # ImageID Series를 따로 변수에 저장합니다.
    image_id = df_list[0]["ImageID"]

    # ImageID column을 제거한 DataFrame을 반환합니다.
    df_list = [df.drop(columns=["ImageID"]) for df in df_list]

    # 첫번째 모델에 가중치 0.9
    df_list[0] *= 0.9

    result = get_submission_df(sum_and_get_argmax(df_list), image_id)
    result.to_csv(os.path.join(
        output_dir, "ensemble/ensemble.csv"), index=False)

    print('Ensemble Done!')
