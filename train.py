from header.Headers import *
import only_plots.plots as plots


def display_errors_dict(err_):
    """
    Display an error message an leave
    :param err_: The list returned from check_data()
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

    # Error Sum of Squares (SSE): Tot((each price - each predicted value)²)
    SSE = sum([(price[i] - prediction[i]) ** 2 for i in range(nbr_data)])
    logger.debug(f'Error Sum of Squares = \x1b[1;30;42m {SSE:.2f} \x1b[0m\n')

    R2 = SSR / (SSR + SSE)
    logger.debug(f'R² = \x1b[1;30;42m {R2:.4f} \x1b[0m\n')

    return R2


def gradient_descent(mileage, price, nbr_data):
    """
    Make the gradient descent
    :param mileage: mileage values
    :param price: mileage values
    :param nbr_data: number of data
    :return: theta0: The new theta0
             theta1: The new theta1
             eval_it: List of all iterations
             eval_theta0: List of all tmp_theta0
             eval_theta1: List of all tmp_theta1
    """

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

    return theta0, theta1, eval_it, eval_theta0, eval_theta1


def compute_and_plot(_file, _args):
    """
    Creates the model and plot results
    :param _file: The checked_file
    :param _args: The args dictionary from argparse
    """

    logger.debug(f'Entering {inspect.currentframe().f_code.co_name}')
    df = pd.read_csv(_file)
    mileage, price = df.loc[:, EXP_COL[0]].values, df.loc[:, EXP_COL[1]].values
    nbr_data = len(mileage)

    # The gradient descent
    theta0, theta1, eval_it, eval_theta0, eval_theta1 = gradient_descent(mileage, price, nbr_data)

    # Predict new values
    prediction = theta0 + theta1 * mileage
    logger.debug(f'\nPredicted values =\n{prediction}\n')

    # Compute metrics
    R2 = compute_metrics(price, prediction, nbr_data)

    # Plots
    plots.set_plots(_args, mileage, price, prediction, eval_it, eval_theta0, eval_theta1, theta0, theta1, logger, R2)

    # Save metrics
    os.makedirs('thetas', exist_ok=True)
    coefs = {'θ0': theta0, 'θ1': theta1}
    df_coefs = pd.DataFrame(coefs.items(), columns=['COEFS_NAMES', 'COEFS'])
    df_coefs.to_excel(os.path.join('thetas', f'coefs.xlsx'))
    logger.debug(f'Saving metrics OK')

    # Print thetas
    logger.debug(f'\n\x1b[1;30;42m θ0 = {theta0:.2f} and θ1 = {theta1:.2f} \x1b[0m\n')


def check_data(_file):
    """
    Check the potential errors in the data set:
        If columns are the one expected => i.e. in EXP_COL
        If columns have a null value
        If columns have a wrong value
        If columns have a negative value
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

    # If a column has a wrong value, a NaN value or a negative value
    for col in act_col:
        for e, val in enumerate(df[col]):
            try:
                # Try wrong value
                val = float(val)

                # Check NaN or negative value
                if np.isnan(val) or val < 0:
                    return 'val', f'value: {val:}, line: {e:}, column: {col:}'
            except ValueError:
                return 'val', f'value: {val:}, line: {e:}, column: {col:}'

    logger.debug(f'File OK')


def parsing():
    """
    Parses and defines parameters
    Creates the logger
    :return: _args, _logger
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
