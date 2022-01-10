
from utils import train_and_hypertune
from mlxtend.evaluate import paired_ttest_kfold_cv, paired_ttest_5x2cv
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



def main() :
    # Train models, load best parameters for each models
    X_train, y_train, svc_best_params, rf_best_params, svc_best_score, rf_best_score = train_and_hypertune()

    # Instantiate models with their params 

    svc = SVC(C=svc_best_params['C'], kernel=svc_best_params['kernel'])
    rf = RandomForestClassifier(
        criterion=rf_best_params['criterion'], n_estimators=rf_best_params['n_estimators'])

    # Statistically compare algorithms using 5x2 cross validation
    t, p = paired_ttest_5x2cv(
        estimator1=svc, estimator2=rf, X=X_train, y=y_train)

    print(f'svc best f1 : {svc_best_score}')
    print(f'rf best f1 {rf_best_score}')
    print('t statistic: %.3f' % t)
    print('p value: %.3f' % p)
    # interpret the result
    if p <= 0.05:
        print('Difference between mean performance is probably real')
    else:
        print('Algorithms probably have the same performance')


if __name__ == '__main__':
    main()

