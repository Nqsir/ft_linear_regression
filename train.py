import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties
import argparse
import os
import sys
import logging
import sklearn.linear_model
EXP_COL = ['km', 'price']


def display_errors_dict(err_):
    dictionary = {
        'col': f'Unexpected column: \x1b[1;37;41m{err_[1]} \x1b[0m\n\n',
        'value': f'Unexpected value: \x1b[1;37;41m {err_[1]} \x1b[0m\n\n',
    }

    print(f'\n{dictionary[err[0]]}')


def check_data(_file):
    df = pd.read_csv(_file)
    act_col = list(df)

    for col in act_col:
        if col not in EXP_COL:
            return 'col', col

    for df_t in df.dtypes:
        if df_t != "float64" and df_t != "int64":
            error = df[~df.applymap(np.isreal).all(1)]
            for it in error:
                print(error[it])
                for e, val in enumerate(error[it]):
                    try:
                        val = float(val)
                    except ValueError:
                        column = str(it)
                        value = str(val)
                        line = e

                        # And construct the line edition to display error
                        wrong = f'value: {value:}, line: {line:}, column: {column:}'
                        return 'value', wrong


def extract_data(_file):
    df = pd.read_csv(_file)
    x_ = df.iloc[:len(df), 0].values
    y_ = df.iloc[:len(df), 1].values

    x_ = x_.reshape(-1, 1)

    regression = sklearn.linear_model.LinearRegression()
    regression.fit(x_, y_)

    print(regression.coef_[0])
    print(regression.intercept_)

    df["predicted"] = regression.predict(x_)

    print(f'test_me = {240000*regression.coef_[0] + regression.intercept_}')
    # print(f'test_le = {regression.predict()}')

    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    fig.set_figheight(5.5)
    fig.set_figwidth(12.5)
    ax1.set_title(f'ft_linear_regression, Reliability : R² = {sklearn.metrics.r2_score(df["predicted"], y_):.4f}',
                  fontsize=14)
    ax1.set_xlabel(f'{EXP_COL[0]}')
    ax1.set_ylabel(f'{EXP_COL[1]}')
    line1, = ax1.plot(x_, y_, 'co', zorder=1, label='Dataset')
    line2, = ax1.plot(x_, df.iloc[:len(df), 2].values, 'r', zorder=1, label='Predicted')
    ax1.grid(linestyle='-', linewidth=1)
    ax1.legend(handles=[line1, line2], loc=1, fontsize=12)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='py train.py')
    parser.add_argument('csv_file', help='A csv file containing data')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode', default=False)
    args = parser.parse_args()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.debug('\n/*/-------\n Debug mode activated \n/*/-------\n')
    args = parser.parse_args()

    file = os.path.join(os.getcwd(), args.csv_file)

    if os.path.exists(file)and os.path.isfile(file) and file.endswith('.csv'):
        err = check_data(file)
        if err:
            sys.exit(display_errors_dict(err))
        extract_data(file)
    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
