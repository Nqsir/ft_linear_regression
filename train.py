import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties
import argparse
import os
import sys
import logging
import sklearn.linear_model
from scipy import stats
import openpyxl
EXP_COL = ['km', 'price']
LEARNING_RATE_THETA0 = 0.01
LEARNING_RATE_THETA1 = 0.0000000000001
CONVERGENCE = 0.01


def display_errors_dict(err_):
    dictionary = {
        'col': f'Unexpected column: \x1b[1;37;41m{err_[1]} \x1b[0m\n\n',
        'val': f'Unexpected value: \x1b[1;37;41m {err_[1]} \x1b[0m\n\n',
        'neg': f'Unexpected negative value: \x1b[1;37;41m {err_[1]} \x1b[0m\n\n',
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
                for e, val in enumerate(error[it]):
                    try:
                        val = float(val)
                    except ValueError:
                        column = str(it)
                        value = str(val)
                        line = e

                        # And construct the line edition to display error
                        wrong = f'value: {value:}, line: {line:}, column: {column:}'
                        return 'val', wrong

    wrong = ''
    for col in act_col:
        for e, v in enumerate(df[col]):
            if float(v) < 0:
                wrong = f'value: {v:}, line: {e:}, column: {col:}'

        if wrong:
            return 'neg', wrong


def extract_data(_file):
    df = pd.read_csv(_file)
    mileage = df.iloc[:, 0].values
    price = df.iloc[:, 1].values
    theta0 = 0
    theta1 = 0
    prev_theta0 = 1
    prev_theta1 = 1
    nbr_data = len(mileage)
    i = 0
    test_it = []
    test_thet0 = []
    test_thet1 = []

    # Ok Working, but the convergence is somehow way to high, need to process some tests on variables, i.e.
    #       LEARNING_RATE_THETA0 = 0.01
    #       LEARNING_RATE_THETA1 = 0.0000000000001
    #       CONVERGENCE = 0.01

    while abs(theta0 - prev_theta0) + abs(theta1 - prev_theta1) > CONVERGENCE:
        prev_theta0 = theta0
        prev_theta1 = theta1
        tmp_theta0 = LEARNING_RATE_THETA0 * (1 / nbr_data) \
                     * sum([(theta0 + theta1 * mileage[i]) - price[i] for i in range(nbr_data)])
        tmp_theta1 = LEARNING_RATE_THETA1 * (1 / nbr_data)\
                     * sum([((theta0 + theta1 * mileage[i]) - price[i]) * mileage[i] for i in range(nbr_data)])
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1
        test_it.append(i)
        test_thet0.append(tmp_theta0)
        test_thet1.append(tmp_theta1)
        i += 1

    fig, ax1 = plt.subplots()
    fig.set_figheight(5.5)
    fig.set_figwidth(12.5)
    line1, = ax1.plot(test_it, test_thet0, 'bo', label='thet0', markersize='2')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Updated Theta0')
    ax2 = ax1.twinx()
    line2, = ax2.plot(test_it, test_thet1, 'ro', label='thet1', markersize='2')
    ax2.set_ylabel('Updated Theta1')
    ax1.legend(handles=[line1, line2], loc=1, fontsize=12)
    plt.show()

    prediction = theta0 + theta1 * mileage
    print(prediction)

    # plt.scatter(mileage, price)
    # plt.plot(mileage, prediction, c='r')
    # plt.show()

    # Meth 1 ------------------------
    slope, intercept, r_value, p_value, std_err = stats.linregress(mileage, price)

    def predict(x):
        return slope * x + intercept

    logger.debug(f'Theta0 = {intercept}')
    logger.debug(f'Theta1 = {slope}')
    logger.debug(f'r**2 = {r_value**2}')

    fitLine = predict(mileage)
    # axes = plt.axes()
    # axes.grid()  # dessiner une grille pour une meilleur lisibilité du graphe
    # plt.scatter(mileage, price)
    # plt.plot(mileage, fitLine, c='r')
    # plt.show()

    print(f'Test diff = {prediction - fitLine}')

    # Meth 2 ------------------------
    # mileage = mileage.reshape(-1, 1)
    # price = price.reshape(-1, 1)
    # regression = sklearn.linear_model.LinearRegression()
    # regression.fit(mileage, price)
    #
    # logger.debug(f'Theta0 = {regression.intercept_[0]}')
    # logger.debug(f'Theta1 = {regression.coef_[0]}')
    # df["predicted"] = regression.predict(mileage)
    #
    # coefs = {'θ0': regression.intercept_[0]}
    # coefs.update(zip([f'θ{e + 1}' for e in range(0, len(regression.coef_[0]))], list(regression.coef_[0])))

    # logger.debug(f'coefs = {coefs}')
    #
    # # Saving coefs and metrics into METRICS_DIR
    # os.makedirs('thetas', exist_ok=True)
    # df_coefs = pd.DataFrame(coefs.items(), columns=['COEFS_NAMES', 'COEFS'])
    # df_coefs.to_excel(os.path.join('thetas', f'coefs.xlsx'))
    #
    # plt.style.use('ggplot')
    # fig, ax1 = plt.subplots()
    # fig.set_figheight(5.5)
    # fig.set_figwidth(7.5)
    # ax1.set_title(f'ft_linear_regression, Reliability : R² = {sklearn.metrics.r2_score(df["predicted"], price):.4f}',
    #               fontsize=14)
    # ax1.set_xlabel(f'{EXP_COL[0]}')
    # ax1.set_ylabel(f'{EXP_COL[1]}')
    # line1, = ax1.plot(mileage, price, 'co', zorder=1, label='Dataset')
    # line2, = ax1.plot(mileage, df.iloc[:len(df), 2].values, 'r', zorder=1, label='Predicted')
    # ax1.grid(linestyle='-', linewidth=1)
    # ax1.legend(handles=[line1, line2], loc=1, fontsize=12)
    # plt.show()


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
