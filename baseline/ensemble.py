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


def load_csv_file(file_loc: str) -> pd.DataFrame:
    """
    csv file을 읽어와 dataframe으로 반환합니다.
    """
    df = pd.read_csv(file_loc)

    return df


def detach_image_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    ImageID column을 제거한 DataFrame을 반환합니다.
    """
    return df.drop(columns=["ImageID"])


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
    parser.add_argument('--data_dir', type=str, default="/outputs")

    args = parser.parse_args()
    data_dirs = args.data_dir

    data_list = load_data_list(data_dirs)
    df_list = [load_csv_file(file_name) for file_name in data_list]
    image_id = df_list[0]["ImageID"]
    df_list = [detach_image_id(df) for df in df_list]
    result = get_submission_df(sum_and_get_argmax(df_list), image_id)

    result.to_csv("submission.csv", index=False)
