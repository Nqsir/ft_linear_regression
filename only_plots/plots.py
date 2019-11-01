from Headers import *


def first_subplot(ax, mileage, price, prediction, theta0, theta1):
    ax[0][0].set_title(f'ft_linear_regression : R² = 0.0000, θ0 = {theta0:.2f}, θ1  = {theta1:.2f}', fontsize=11)
    ax[0][0].set_xlabel(f'{EXP_COL[0]}')
    ax[0][0].set_ylabel(f'{EXP_COL[1]}')
    line_0_0_1, = ax[0][0].plot(mileage, price, 'co', zorder=1, label='Dataset')
    line_0_0_2, = ax[0][0].plot(mileage, prediction, 'r', zorder=1, label='Predicted')
    ax[0][0].grid(linestyle='-', linewidth=0.5)
    ax[0][0].legend(handles=[line_0_0_1, line_0_0_2], loc=0, fontsize=11)


def second_subplot(ax, eval_it, eval_theta0, eval_theta1):
    ax[0][1].set_title(f'Gradient descent convergence', fontsize=11)
    ax[0][1].set_xlabel('Iterations')
    ax[0][1].set_ylabel('Updated Theta0')
    line_0_1_1, = ax[0][1].plot(eval_it, eval_theta0, 'b', label='theta0', markersize='0.9', zorder=2)
    ax_0_1_2 = ax[0][1].twinx()
    line_0_1_2, = ax_0_1_2.plot(eval_it, eval_theta1, 'g', label='thet1', markersize='0.9', zorder=1)
    ax_0_1_2.set_ylabel('Updated Theta1')
    ax[0][1].grid(linestyle='-', linewidth=0.5, zorder=3)
    ax_0_1_2.legend(handles=[line_0_1_1, line_0_1_2], loc=0, fontsize=11)


def third_subplot(ax, mileage, price):
    import sklearn.linear_model
    regr = sklearn.linear_model.LinearRegression()
    _mileage = mileage.reshape(-1, 1)
    _price = price.reshape(-1, 1)
    regr.fit(_mileage, _price)
    predict = regr.predict(_mileage)
    ax[1][0].set_title(f'sklearn.linear_model.LinearRegression() : R² = {sklearn.metrics.r2_score(predict, price):.4f},'
                       f'θ0 = {regr.intercept_[0]:.2f}, θ1  = {regr.coef_[0][0]:.2f}', fontsize=8)
    ax[1][0].set_xlabel(f'{EXP_COL[0]}')
    ax[1][0].set_ylabel(f'{EXP_COL[1]}')
    line_1_0_1, = ax[1][0].plot(mileage, price, 'co', zorder=1, label='Dataset')
    line_1_0_2, = ax[1][0].plot(mileage, predict, 'r', zorder=1, label='Predicted')
    ax[1][0].grid(linestyle='-', linewidth=0.5)
    ax[1][0].legend(handles=[line_1_0_1, line_1_0_2], loc=0, fontsize=11)


def fourth_subplot(ax, mileage, price, prediction, logger):
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(mileage, price)

    fit_scipy = slope * mileage + intercept
    ax[1][1].set_title(f'scipy.stats.lineregress() : R² = {r_value ** 2:.4f}, θ0 = {intercept:.2f}, θ1  = {slope:.2f}',
                       fontsize=10)
    ax[1][1].set_xlabel(f'{EXP_COL[0]}')
    ax[1][1].set_ylabel(f'{EXP_COL[1]}')
    line_1_1_1, = ax[1][1].plot(mileage, price, 'co', zorder=1, label='Dataset')
    line_1_1_2, = ax[1][1].plot(mileage, fit_scipy, 'r', zorder=1, label='Predicted')
    ax[1][1].grid(linestyle='-', linewidth=0.5)
    ax[1][1].legend(handles=[line_1_1_1, line_1_1_2], loc=0, fontsize=11)

    logger.debug(f'Test diff = {prediction - fit_scipy}\n')


def set_plots(mileage, price, prediction, eval_it, eval_theta0, eval_theta1, theta0, theta1, logger):
    # Set figure
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[12, 7])

    # Set subplots
    first_subplot(ax, mileage, price, prediction, theta0, theta1)
    second_subplot(ax, eval_it, eval_theta0, eval_theta1)
    third_subplot(ax, mileage, price)
    fourth_subplot(ax, mileage, price, prediction, logger)


def set_gd_plot_only(mileage, price, prediction):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7.5, 5])
    ax.set_title(f'ft_linear_regression, Reliability : R² = To be continued',
                 fontsize=14)
    ax.set_xlabel(f'{EXP_COL[0]}')
    ax.set_ylabel(f'{EXP_COL[1]}')
    line1, = ax.plot(mileage, price, 'co', zorder=1, label='Dataset')
    line2, = ax.plot(mileage, prediction, 'r', zorder=1, label='Predicted')
    ax.grid(linestyle='-', linewidth=1)
    ax.legend(handles=[line1, line2], loc=1, fontsize=12)
