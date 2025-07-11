# rq3_long_mean_std_with_label.py
import pandas as pd
from pathlib import Path
from pandas.api.types import CategoricalDtype


ALG_ORDER = ['Transfer-GA', 'Vanilla-GA']


def wide_to_long(csv_path: Path) -> pd.DataFrame:

    system = csv_path.stem.split('_')[0]

    df = pd.read_csv(csv_path)


    alg_cols = [c for c in ALG_ORDER if c in df.columns]
    if not alg_cols:
        raise ValueError(f'{csv_path} cannot find algorithm column {ALG_ORDER}')

    long_df = (
        df.melt(
            id_vars=['Environment', 'Label'],
            value_vars=alg_cols,
            var_name='Algorithm',
            value_name='Score'
        )
    )

    long_df.rename(columns={'Environment': 'Workload'}, inplace=True)
    long_df.insert(0, 'System', system)

    algo_cat = CategoricalDtype(ALG_ORDER, ordered=True)
    long_df['Algorithm'] = long_df['Algorithm'].astype(algo_cat)
    long_df.sort_values(['Workload', 'Algorithm'], inplace=True)
    long_df.reset_index(drop=True, inplace=True)

    return long_df[['System', 'Workload', 'Algorithm', 'Score', 'Label']]


def main(src_dir='insight5', dst_dir='insight5_format'):
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(exist_ok=True)

    for csv_file in src.glob('*_Transfer-GA_vs_Vanilla-GA.csv'):
        long_df = wide_to_long(csv_file)
        out_path = dst / f'{long_df.at[0, "System"]}.csv'
        long_df.to_csv(out_path, index=False)
        print(f'✅  {csv_file.name} → {out_path.name}')


if __name__ == '__main__':
    main()
