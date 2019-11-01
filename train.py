from Headers import *
import only_plots.plots as plots


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


def compute(_file, _args):
    df = pd.read_csv(_file)
    mileage, price = df.iloc[:, 0].values, df.iloc[:, 1].values
    nbr_data = len(mileage)
    theta0, theta1 = 0, 0
    prev_theta0, prev_theta1 = 1, 1
    i = 0
    eval_it, eval_theta0, eval_theta1 = [], [], []

    while abs(theta0 - prev_theta0) + abs(theta1 - prev_theta1) > CONVERGENCE:
        prev_theta0 = theta0
        prev_theta1 = theta1
        tmp_theta0 = LEARNING_RATE_THETA0 * (1 / nbr_data) \
                     * sum([(theta0 + theta1 * mileage[i]) - price[i] for i in range(nbr_data)])
        tmp_theta1 = LEARNING_RATE_THETA1 * (1 / nbr_data)\
                     * sum([((theta0 + theta1 * mileage[i]) - price[i]) * mileage[i] for i in range(nbr_data)])
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1
        if args.evaluate:
            eval_it.append(i)
            eval_theta0.append(tmp_theta0)
            eval_theta1.append(tmp_theta1)
            i += 1

    prediction = theta0 + theta1 * mileage
    logger.debug(f'Predicted values = {prediction}\n')
    # logger.debug(f'r**2 = {r_value**2}') => need to def

    plt.style.use('ggplot')
    if args.evaluate:
        plots.set_plots(mileage, price, prediction, eval_it, eval_theta0, eval_theta1, theta0, theta1, logger)
    else:
        plots.set_gd_plot_only(mileage, price, prediction)

    plt.tight_layout()
    plt.show()
    plt.clf()

    # Saving metrics
    os.makedirs('thetas', exist_ok=True)
    coefs = {'θ0': theta0, 'θ1': theta1}
    df_coefs = pd.DataFrame(coefs.items(), columns=['COEFS_NAMES', 'COEFS'])
    df_coefs.to_excel(os.path.join('thetas', f'coefs.xlsx'))

    logger.debug(f'θ0 = {theta0} & θ1 = {theta1}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='py train.py')
    parser.add_argument('csv_file', help='A csv file containing data')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode', default=False)
    parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate the model', default=False)
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
        compute(file, args)
    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
