# Linear Regression

> The Algorithm who serves as a good big jump off point

## pre-start

- the data we will use, The Advertising data set, have 4 columns :

- 1 - Tv

- 2 - Radio

- 3 - Newspaper

- 4 - Sales ( Target)

![Alt text](image.png)

### Important Question

1- Is there a relationship between advertising budget (X) and sales(Y)?

2- How strong is the relationship between advertising budget (X) and sales (Y)?

3- Which media are associated with sales?

4- How large is the association between each medium and sales?

5- How accurately can we predict future sales?

6- Is the relationship linear?

7- Is there synergy among the advertising media? : -

- Perhaps spending $50,000 on television advertising and $50,000 on radio advertising is associated with higher sales than allocating $100,000 to either television or radio individually

## 1.1 Simple Linear Regression

- the population function :

$\ ` sales ≈ β0  + β1 × TV `$

$\ ` β_0 `$ : interception with y-axis ( **Unknown )**

$\ ` β_1 `$ : Slope **( Unknown )**

and we want to **estimate the parameters** ( $\ `βˆ_0 and βˆ_1`$)to get the fit line where we can predict the future.

so our function is $\ `yˆ = β_ˆ0  + β_ˆ1  * x,`$

### 1.1.1 Estimating the Coefcients

There are a number of ways of measuring closeness. However, by far the most common approach involves minimizing the least squares criterion.

$e = y_i - yˆ_i$

$RSS = e^2_1  + e^2_2  + ··· + e^2_n$

- The least squares approach chooses $βˆ0$ and $βˆ1$ to minimize the RSS. Using some calculus, one can show that the minimizers are,

$βˆ_1 = \frac{\sum_{i=1}^n (xi − x¯)(yi − y¯)}{\sum_{i=1}^n(x_i - x_¯)}$, <br>

$βˆ_0 = y¯ − βˆ1  * x$

![Alt text](image-1.png)

--- $βˆ0 = 7.03$ and, ---$βˆ1 = 0.0475.$

### 1.1.2 Assessing the Accuracy of the Coefcient Estimates

If f(X) is to be approximated by a linear function, then we can write this relationship as $Y = β0  + β1  * X + ε$ for the population function not the

sample function will explain right away

**where ε : is a mean-zero random error term**

![Alt text](image-2.png)

The analogy between linear regression and estimation of the mean of a random variable is an apt one based on the concept of bias,

µˆ might overestimate µ, and on the basis of another set of observations,

µˆ might underestimate µ. But if we could average a huge number of estimates of µ obtained from a huge number of sets of observations,

then this average would exactly equal µ.

is pretty close to the true population regression line.

a big question here, how accurate is the sample mean µˆ as an estimate of population mean µ?

so as we see we need a thing to tell us,

here the standard error will do it for us,

![Alt text](image-4.png)

In general, $σ^2$ is not known, but can be estimated from the data. This estimate of σ is known as the residual standard error,

and the formula is $RSE = \sqrt  {RSS/(n − 2)} = σ^2$,

H0 : There is no relationship between X and Y

versus

Ha : There is some relationship between X and Y .

Mathematically, this corresponds to testing

H0 : β1 = 0

versus

Ha : β1 ≠ 0,

In practice, we compute a t-statistic, t-statistic given by,

$t = \frac{\hat{β_1} - 0}{SE(\hat{β_1} )}$

which measures the number of standard deviations that βˆ1 is away from 0.\*\*

---

### 1.1.3 Assessing The Accuracy Of The Model

![Alt text](image-5.png)

#### Residual Standard Error

RSE : squer_root(RSS) > that is standard daviation

1- use to see how much far

2- lack of fit

---

#### $R-Square$

The RSE provides an absolute measure of lack of ft of the to the data.

But since it is measured in the units of Y , it is not always clear what constitutes a good RSE.

The R2 statistic provides an alternative measure of ft.

$ٌR2 = 1  -  \frac{RSS}{TSS}$

where, $TSS = \sum_{i=1}^n {(y_i - y¯)  ^  2}$

R2 measures the proportion of variability in Y that can be explained using X. An R2 statistic that is close to 1 indicates that a large proportion of the variability in the response is explained by the regression. A number near 0 indicates that the regression does not explain much of the variability in the response

**the other columns statistics**

![Alt text](image-6.png)

# 2. Multiple Linear Regression

Instead of fitting a separate simple linear regression model for each predictor,

a better approach is to extend the simple linear regression model, so that it can directly accommodate multiple predictors, we can do this by giving each predictor a separate slope coefficient in a single model.

$Y= β_0  + β_1  * X_1  + β_2  * X_2  + ... + β_n * X_n + ε$

$Sales = β_0  + β_1  * TV + β_2  * Newspaper + β_n * Radio + ε$

but as we know the coefficients is unknown, and must be estimated

$\hat{y} = \hat{β_0}  +  \hat{β_1}  * x_1  +  \hat{β_2}  * x_2  + ... +  \hat{β_n}  * x_n + ε$

![Alt text](image-3.png)

and there are also relationship between predictors and each other,

![Alt text](image-8.png)

![Alt text](image-7.png)

that the correlation between radio and newspaper is 0.35,
the markets with high newspaper adv, tend to also have high radio adv.

we suppose that the multiple regression is correct and
newspaper advertising is not associated with sales, but radio advertising
is associated with sales. Then in markets where we spend more on radio
our sales will tend to be higher, and as our correlation matrix shows, we
also tend to spend more on newspaper advertising in those same markets.
Hence, in a simple linear regression which only examines sales versus
newspaper, we will observe that higher values of newspaper tend to be associated
with higher values of sales, even though newspaper advertising is
not directly associated with sales. So newspaper advertising is a surrogate
for radio advertising; newspaper gets “credit” for the association between
radio on sales, and it dosen't improve the prediction.

## 2.2 Some Important questions

    1- Is at least one preds, useful in prediction?

    2- Do all preds, help in explaion Y, or only subset do?

    3- How will the model fit the data ?

    4- How accurate is our prediction ?

### 1- is there relation between preditors and response?

we can simply check whether $ β_i = 0$ but for all.

we will use the test hypothesis.

> $H_0 : β_i = 0$ for all

> versus

> $H_a : β_i ≠ 0,$ at least one

and this hypothesis is performed by computing the F-static,

$F = \frac{(TSS - RSS) / p}{RSS / (n - p - 1)}$

if $RSS / (n - p - 1) = \alpha{}^2$ that provided $H_0$ is true,

if $(TSS - RSS) / p = \alpha{}^2$ that provided $H_0$ is true,

when there is no relation, F is close to 1.

if $H_a$ is True, then $(TSS - RSS) / p > \alpha{^2}$

so we expect that F > 1.

### 2- Deciding in Important Variables?

is referred to as variable selection.

we would like to perform this method by trying out a lot of diff models, each containing a diff subset of predictor,
for instance if p = 2, so we can consider 4 models,

    1- no var
    2- X1 only
    3- X1, and X2
    4- X2 only

and then choose the best model.<br>
**unfortunatley** there are $2^p$ model to compare, not practical.

there are three classical approches for this task :

- **Forward Selection** : begin with null model only intercept, then we fit p simple linear regreion and add to null model that the result in the RSS is lowesr, until some stopping rule is satified.

- **Backward Selection** : begin will all var, and then remove the largest p-value

- **Mixed Selection** : Mixed of **forward and backward**,
  begin as forward with null and continue when p-value is rises as high remove the varieble from the model.

we cannot use backward if p > n, while forward alwayes can be used becuse is greedy approch.

### 3 - model fit :

Two of the most common numerical measures of model fit are the RSE and
R2, the fraction of variance explained. These quantities are computed and
interpreted in the same fashion as for simple linear regression.

It turns out that R2 will always
increase when more variables are added to the model, even if those variables,
are only weakly associated with the response.

so we will use the RSE where the most accurate is the lowest one,

$RSE = \sqrt{ \frac{1}{ n - p - 1} * RSS}$

where p : is number of predictor variables

### 4 - Predictions:

- we estimate the coff, and is only estimate for the truth,
  we can compute a confidance interval in order to detrmine how close
  $\hat{Y}$ will be to $f(x)$

- model bias for reducible error, we work assum that the model is correct
  and ignore this discrepancy.
