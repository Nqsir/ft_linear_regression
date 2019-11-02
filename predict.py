from Headers import *

if __name__ == '__main__':
    file = os.path.join(os.getcwd(), 'thetas\coefs.xlsx')
    if os.path.exists(file) and os.path.isfile(file) and file.endswith('.xlsx'):
        df = pd.read_excel(file)
        theta0 = df.iloc[0, 2]
        theta1 = df.iloc[1, 2]
        try:
            theta0 = float(theta0)
            theta1 = float(theta1)
        except ValueError:
            sys.exit(f'\n\x1b[1;37;41m Wrong thetas \x1b[0m\n')
    else:
        print('thetas have not been computed, set to default')
        theta0, theta1 = 0, 0

    try:
        while True:
            in_put = input('Enter mileage: ')
            try:
                in_put = float(in_put)
                if in_put < 0:
                    sys.exit(f'\n\x1b[1;37;41m Wrong mileage \x1b[0m\n')
                else:
                    sys.exit(print(f'\n\x1b[1;30;42m The estimated price is: {theta0 + theta1 * in_put:.2f} \x1b[0m\n'))
            except ValueError:
                sys.exit(f'\n\x1b[1;37;41m Wrong mileage \x1b[0m\n')
    except KeyboardInterrupt:
        sys.exit(print('\n\n(┬┬﹏┬┬)\n'))
