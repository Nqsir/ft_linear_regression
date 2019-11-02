from header.Headers import *
import only_plots.plots as plots


def display_errors_dict(err_):
    """
    Display an error message an leave
    :param err_: The list returned from check_data()
    :return:
    """
    logger.debug(f'Entering {inspect.currentframe().f_code.co_name}')
    dictionary = {
        'col': f'Unexpected column: \x1b[1;37;41m{err_[1]} \x1b[0m\n\n',
        'val': f'Unexpected value: \x1b[1;37;41m {err_[1]} \x1b[0m\n\n',
        'neg': f'Unexpected negative value: \x1b[1;37;41m {err_[1]} \x1b[0m\n\n',
        'nan': f'Unexpected NaN value: \x1b[1;37;41m {err_[1]} \x1b[0m\n\n',
    }
    print(f'\n{dictionary[err[0]]}')


def compute_metrics(price, prediction, nbr_data):
    """
    Compute the metrics to get the Coefficient of determination
    :param price: price data
    :param prediction: predicted data
    :param nbr_data: number of data
    :return: R2
    """

    # Metrics
    p_mean = sum(price) / len(price)
    logger.debug(f'p_mean = \x1b[1;30;42m {p_mean:.2f} \x1b[0m\n')

    # Sum of Squares due to Regression (SSR): Tot((each predicted value - price_mean)²)
    SSR = sum([(prediction[i] - p_mean) ** 2 for i in range(nbr_data)])
    logger.debug(f'\nSum of Squares due to Regression = \x1b[1;30;42m {SSR:.2f} \x1b[0m')

    # Total Sum of Squares (SST): Tot((each price - each predicted value)²)
    SST = sum([(price[i] - prediction[i]) ** 2 for i in range(nbr_data)])
    logger.debug(f'Total Sum of Squares = \x1b[1;30;42m {SST:.2f} \x1b[0m\n')

    R2 = SSR / (SST + SSR)
    logger.debug(f'R² = \x1b[1;30;42m {R2:.4f} \x1b[0m\n')

    return R2


def compute_and_plot(_file, _args):
    """
    Creates the model and plot results
    :param _file: The checked_file
    :param _args: The args dictionary from argparse
    """
    logger.debug(f'Entering {inspect.currentframe().f_code.co_name}')
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
    logger.debug(f'\nPredicted values = {prediction}\n')

    # Compute metrics
    R2 = compute_metrics(price, prediction, nbr_data)

    # Plotting part
    plt.style.use('ggplot')
    if args.evaluate:
        plots.set_plots(mileage, price, prediction, eval_it, eval_theta0, eval_theta1, theta0, theta1, logger, R2)
    else:
        plots.set_gd_plot_only(mileage, price, prediction, logger, R2)
    plt.tight_layout()
    plt.show()
    plt.clf()

    # Saving metrics
    os.makedirs('thetas', exist_ok=True)
    coefs = {'θ0': theta0, 'θ1': theta1}
    df_coefs = pd.DataFrame(coefs.items(), columns=['COEFS_NAMES', 'COEFS'])
    df_coefs.to_excel(os.path.join('thetas', f'coefs.xlsx'))
    logger.debug(f'Saving metrics OK')

    # Print thetas
    logger.debug(f'\n\x1b[1;30;42m θ0 = {theta0:.2f} and θ1 = {theta1:.2f} \x1b[0m\n')


def check_data(_file):
    """
    Check the potential errors in the data set
    :param _file: A csv file
    :return: list, first element is the dictionary to raise an error, second is information to display
    """
    logger.debug(f'Entering {inspect.currentframe().f_code.co_name}')
    df = pd.read_csv(_file)
    act_col = list(df)

    # If columns are not the ones expected
    for col in act_col:
        if col not in EXP_COL:
            return 'col', col

    # If a column has a NaN value
    errors = df.isnull().any()
    list_err = [key for key in errors.keys()]
    for e, err in enumerate(errors):
        if err is True:
            # Saving errors to display the line and column concerned
            column = list_err[e]
            df_values = df[df[column].isnull()]
            line = line = int(df_values[column].keys()[0])
            value = 'Null'

            # And construct the line edition to display error
            wrong = f'value: {value:}, line: {line:}, column: {column:}'
            return 'nan', wrong

    # If a column has a wrong value
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

    # If a column has a negative value
    wrong = ''
    for col in act_col:
        for e, v in enumerate(df[col]):
            if float(v) < 0:
                wrong = f'value: {v:}, line: {e:}, column: {col:}'
        if wrong:
            return 'neg', wrong

    logger.debug(f'File OK')


def parsing():
    """
    Parses and defines parameters
    Creates the logger
    :return: _args and _logger
    """
    parser = argparse.ArgumentParser(prog='py train.py')
    parser.add_argument('csv_file', help='A csv file containing data')
    parser.add_argument('-d', '--details', action='store_true', help='Detailed mode', default=False)
    parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate the model', default=False)
    _args = parser.parse_args()
    logging.getLogger('matplotlib.font_manager').disabled = True
    _logger = logging.getLogger()
    if _args.details:
        _logger.setLevel(logging.DEBUG)
    else:
        _logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    _logger.addHandler(stream_handler)
    _logger.debug('\n/*/-------\n Detailed mode activated \n/*/-------\n')
    return _args, _logger


if __name__ == '__main__':
    # Parse arguments and create the logger
    args, logger = parsing()
    logger.debug(f'Logger created in {inspect.currentframe().f_code.co_name}')

    file = os.path.join(os.getcwd(), args.csv_file)
    if os.path.exists(file)and os.path.isfile(file) and file.endswith('.csv'):

        # Check the dataset
        err = check_data(file)
        if err:
            sys.exit(display_errors_dict(err))

        # Compute thetas, metrics and plot
        compute_and_plot(file, args)
        logger.debug(f'Leaving {inspect.currentframe().f_code.co_name}')
    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
