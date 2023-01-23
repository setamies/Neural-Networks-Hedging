**Deep Hedging Fixed Strike Asian Options**

*Lauri Teerimäki*

*Yonsei Univesity, Seoul, South Korea*

**ABSTRACT**

Controlling ones risk in the financial markets is something traders have been trying to master for decades. In 1973, Black & Scholes realized that two risky positions taken together, could essentially eliminate risk itself, which resulted in the creation of the Black-Scholes model, one of the most important concepts in financial theory. However, the model makes some heavy assumptions, such as no trading costs and continuous rebalancing, which make the model suboptimal in real use. This paper uses deep hedging, a method based on neural networks, for hedging arithmetic average Asian fixed strike options. Arithmetic average Asian fixed strike options are path dependent options that do not have a closed form equation, and hence do not have a delta. A pseudo-delta will be estimated by the neural network model.

**1 Introduction**

This paper aims to hedge arithmetic average Asian fixed strike options using deep neural networks. The Black-Scholes model is one of the most important and widely used models in modern finance, but it makes some heavy assumptions that make it suboptimal for pricing options in real life. Essentially the model assumes perfect markets where there are no transaction costs. This leaves it for the traders to take into consideration the market imperfections. Deep hedging uses neural networks to estimate a pseudo-delta and to minimize the cost of hedging. In this paper, a semi- recurrent neural network is built and is tested for hedging arithmetic average Asian fixed strike options. 

Asian options are path dependent options where the options price relies on the average price of the underlying asset between a certain period, which creates some interesting implications. First and foremost, the options are highly path dependent, which make them more difficult to evaluate correctly. They also have a lower volatility than their traditional counterparts, which make them intriguing for hedging purposes and being less exposed to market manipulation.

**2 Theoretical framework**

1. **Black – Scholes model**

The Black-Scholes model is one of the quintessential models in financial theory. The model is used for calculating the theoretical value of options. Since almost all corporate liabilities can be regarded as combinations of options, the model is also applicable to corporate liabilities such as common stocks, warrants and corporate bonds (Black & Scholes 1973). 

The following Stochastic Differential Equation can be used to describe asset prices:

*dS*=σ *SdZ*+(μ−δ)*Sdt*   (1)

Where S is the asset value, σ is the volatility, μ is the drift or expected return, δ is the continuous dividend yield of the asset, dS represents incremental changes in asset value, dZ is a Weiner process and dt represents incremental changes in time. Lakhlani (2013) derives the Black-Scholes model in his research paper by using Ito’s Lemma.

By applying Ito’s Lemma for δ = 0, we can notice that S solves the aforementioned equation. Then V(S,t) solves the following:

*dV*= σ*SV dX*+(μ*SV* +1σ2*S*2*V* +*V* )*dt*    (2)

*s s* 2 *SS t*

By constructing a portfolio ∏, with one long option V(S,t) and being short a Δ fraction of the underlying asset

Π=*V*−Δ⋅*S*⇒ *d*Π=*dV*−Δ*dS*     (3), (4)

and substituting equations 1 and 2 into 4, and after some algebraic manipulation, the following formula can be derived:

*d*Π=σ *S*(*V* − Δ)*dX*+(μ*SV* +1σ2*S*2*V* +*V* −μΔ *S*)*dt*

*S S* 2 *SS t*

for  Δ=*V* ,we get

*S*

*d*Π=(μ*SV* +1 σ2*S*2*V* +*V* )*dt*

*S* 2 *SS t*

By constructing the portfolio this way, the portfolio is completely independent of the randomness from the underlying asset. One of the assumptions in the Black-Scholes model is a no riskless arbitrage market. Assuming there are no riskless arbitrage opportunities and by additionally investing and divesting at a risk-free-rate, the equation above turns into

Π= 1(μ*SV* +1 σ2*S*2*V* +*V* )    (5)

*r S* 2 *SS t*

Now, substituting *Δ= Vs* and (5) in (3) the Black-Scholes Equation (BSE) can be shown as

*v* +1 σ2*S*2*V* +*rSV* − *rV*=0

*t* 2 *SS s*   (6)

In the derivation above, the Black-Scholes model makes several different assumptions. Black and Scholes assume that the underlying asset follows geometric Brownian motion with constant volatility, that there are no dividends and that there is a constant risk-free rate, which market participants can use to borrow or lend. Additionally, they assume that:

1. The number of outstanding stocks is constant
1. The price of the stock is log-normally distributed
1. No transaction costs 

The payoff of a European call option at the time of maturity is:

*C = max(S-K, 0)*

For a strike price, *K > 0, max(0,0-K) = 0.* The underlying asset growing without a bound, the payoff will be *max(0, S-K) = S*. Hence, the boundary conditions are:

*C*(0*,t*)=0 *C*(*S*→∞ *,t*)=*S*

Using the boundary conditions in BSE, the equation for a non-dividend paying European call option can be derived as: 

*C*=*S N*(*d* )− *K* e−*rT N*(*d* )   (6)

0 1 2

The payoff for a European put option can be expressed with:

*P*= *Ke*−*rTN* (−*d*2)− *S N*(−*d* )   (7)

0 1

where S0 is the initial value of the stock price, K is the strike price, T is the maturity, *r* is the risk- free interest rate and N() is the cumulative standard normal distribution function. Furthermore, d

1 and d are calculated as 

2

*S* 1

ln( 0)+(*r*+ σ2)*T*

= *K* 2![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.001.png)

*d*1 σ√*T*

*d* =*d* − σ√*T![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.002.png)*

2 1

where d can be interpreted as the option delta and d as the risk neutral probability of the option

1 2

being exercised. 

2. **Asian options**

Asian options are options where the underlying variable is the average price over a fixed period of time. The payoff of the option therefore depends on the average price of the asset under a certain time, instead of the price when the option expires. There are several different kinds of Asian options. The key differentiating characteristics between them are when they can be exercised, is the option fixed- or floating strike, and the way the average is calculated. This paper will solely focus on European-style Asian fixed strike options. Both averaging methods are discussed in the paper.

In European options, the payoff of the option at the date of maturity is the difference in the current spot price of the underlying asset and the strike price, whereas the payoff in Asian options is determined by the difference of the strike price and the average value of the underlying asset over a fixed time interval (Zhang 2009). This feature leads to certain interesting implications. By counting the difference between the strike price and the average value of the underlying asset, the payoff becomes less volatile compared to more traditional options contracts, such as European options. The reduced volatility also reduces the risk of market manipulation of the underlying instrument at maturity, as manipulating the price of the underlying asset near the maturity date would be less impactful than it would be with options that only use the spot price when the contract is exercised (Rogers & Shi 1995). 

In general, the lower volatility makes Asian options less expensive than their counterparts. Asian options can be useful in delta hedging, as the delta converges to 0 as the contract approaches maturity*.* 

*F**igure 1: Payoff diagram of a long call option***

![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.003.png)

1. **Asian fixed strike option payoffs**

The payoff diagram in figure 1 works both for Asian options as well as it’s traditional counterparts. Recall that the payoff of a European call option can be expressed with the following formula: 

*C*=*max*(*S* − *K,*0)

*T*

The payoff of an Asian fixed strike long call option can be expressed as 

*C*=*max*(*S*¯− *K,*0)

*T*

where  *S*¯ is the average price of the underlying asset under a fixed time period at the date of the

*T*

maturity. The following section will explain the payoffs in greater detail.

The payoffs for the arithmetic average Asian options are calculated as

*C*=*max*( 1∫*T S*(*t*)*dt*− *K ,*0) or  *P*=*max*(*K*− 1∫*T S*(*t*)*dt ,*0)      (8), (9)

*T T*

0 0

The continuous geometric average Asian option with:

(1∫*T* log(*s*)*dt*) (1∫*T* log(*s*)*dt*)

*C*=*max*(*e T* 0 − *K,*0) or  *P*=*max*(*K*− *e T* 0 *,*0)  (11), (12) The discrete arithmetic average Asian option with

*C*=*max*( 1 ∑*m S*(*t* )− *S*(*t* )*,*0)  

*m* =0 *i K ,*0) or  *P*=*max*(*K*− 1 ∑*m i* (13), (14)

*m*

*i i*=0

The disrcrete geometric average Asian option with

*C*=*max*(*eT*1∑*m* log(*S*(*ti*)¿)− *K ,*0) or  *P*=*max*(*K*− *eT* 0 *,*0)  (15), (16)

*m*

1∑ log(*S*(*t* )¿)

*i*

0

2. **Pricing Asian fixed strike options**

We assume that the price of the underlying asset follows a log-normal distribution continuous in time. The product of log-normally distributed values  is also log-normal, while the sum is not. Hence, it is possible to use the derive the Black-Scholes model to a closed form analytic solution. The arithmetic averaging method doesn’t follow a log-normal distribution, which is why there are no closed form equations to price Asian options using arithmetic averaging (Zhang. H, 2009). 

1. ***Geometric closed form solution***

Following the closed form equation by Kemna and Vorst (1990) where the geometric averaging pricing solution alters the volatility and cost of carry term, the closed form equation for an Asian call and put option can be given as:

*C*=*S eaN* (*d* )− *KecN*(*d* ) or  *P*= *KecN*(−*d* )− *S eaN*(−*d* ) (17), (18)

*t* 1 2 2 *t* 1

Where N() is the cumulative standard normal distribution function

*a*=*b*(*b*−*r*)(*T*−*T* )

0

1(*r*− *D*− σ2

*b*= )

2 6

*c*=−*r*(*T*−*T* )

0

where r is the risk free rate, T is the maturity date and T is the beginning date. 

0

(*S* )

ln 0~~ +(*b*− 1( σ~~ )2)*T![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.004.png)*

*d* = *K* σ 2 √3 − σ √![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.005.png)![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.006.png)![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.007.png)![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.008.png)

1 √*T*~~ ,  *d*2= *d*1 √3 *T*

√3

2. ***Pricing an arithmetic Asian option***

As it was stated previously, the sum of log-normally distributed values is not log-normal and therefore there are no closed form solutions for pricing Asian options with arithmetic averaging. There are several different methods for trying to calculate the price of arithmetic Asian options. Hull and White (1993) applied binomial option pricing method by Cox, Ross & Rubinstein (1979) both for Eurasian and Amerasian options. Boyle (1977) was the first person to apply Monte Carlo simulations to financial derivatives. Monte Carlo methods have a fascinating history behind them, being used to simulate the behaviour concerning neural diffusion in fissionable material, among other things. The Monte Carlo method can be used not only for solving deterministic problems, deterministic problems can be solved by the Monte Carlo method, given it has the same formal expression as some stochastic process  (Galda 2008). Monte Carlo simulation is highly robust and works well for solving multi-dimensional or path-deterministic problems, which is why it is also applicable to pricing Asian options. Monte Carlo simulation heavily relies on the law of large numbers, which in this context implies how obtaining a large number of results from trials, and taking the average value of them, should be close to the expected value and tends to become more accurate the more trials are performed. This can be mathematically expressed as:

lim∑*n Xi*= ¯*X n*→∞ *i*=1 *n*

Unfortunately, using Monte Carlo simulations can be highly time consuming due to the large amount of estimations required, which becomes especially devious if the estimations are complex to calculate. This paper will use Monte Carlo methods to price Asian fixed strike options, as well as use a control variate method in order to reduce the variance and get quicker results.

3. **Monte Carlo Simulation** 

The Monte Carlo simulation will go as follows:

- Sample a random price path for S following Geometric Brownian Motion, under a risk neutral

probability

- Calculate the payoff for the derivative per simulation
- Repeat n times to gain a sufficient sample
- Compute the mean of the derivative payoffs 
- In order to gain the estimate of the value of the derivative, discount the mean with the risk- free rate

geometric Brownian motion can be expressed as 

*S* =*S e*(*r*− σ2 )*t*+σ *Wt*

2

*t* 0

*Figure 2: Asset price simulations following GBM, low volatility case (std = 0.2)*

[^1]*Figure 3: Asset price simulations following GBM, high volatility case (std = 0.6)*

![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.009.png)

First, the payoff of an Asian arithmetic fixed strike call option is calculated with equation 13, with S0 and K both being 100, r = 0, σ = 0.2, time to maturity 30/365 and 30 simulated steps, the option price began stabilizing after 1000 simulations, with a standard deviation of 1.93. Running 100 000 simulations yielded a call option price of roughly 1.313, with a standard deviation of 1.957. This leaves a lot of room for improvement, which is why a control variate method was applied to the arithmetic option pricing. 

Following the research paper by Kemna & Vorst, a control variate variance reduction method was constructed for the Monte Carlo simulations. If the desired simulation quantity is *θ = E(X),* and there is another random variable Y, with a known expectation of *μɣ = E(Y),* there is an unbiased estimator of θ, *Z = X – c(Y – μɣ),* by linearity of expectation. It can be proved that

*V*(*Z*)=*V* (*X*)+*c*2*V*(*Y*)+2*cCov*(*X,Y*)

*Cov*(*X ,Y*)2

is minimized when  *c*=−~~ . Here, Y is the control variate of X. To reduce variance, a

*V* (*Y*)

*Y* with a high correlation to *X* is chosen. In this case, *X* is the Asian option with arithmetic sampling, while *Y*  is  the Asian  option  with  geometric  sampling.  Provided  that *Y*  is  a  satisfactory approximation of *X*, the standard deviation should reduce drastically. 

The control variate is calculated with the following equations:

*N*

*X* =*e*−*rT*[*max*( 1 ∑ *S* − *K ,*0)]

[^2] *N i*

*i*=0

∑*N* (*X* − ¯*X*)(*Y* −*Y*¯)

*i i*

*c*=− *i*=0 ∑*N* (*X* − *X*¯)2

*i*

*i*=0

(ln(*S*0)+(*r*+σ2)*T* )

*d*= *K ![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.010.png)* 6 2

σ√*T*

3![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.011.png)

μγ= *e*− *rT*(*e*φ*S N*(*d*)− *KN*(*d*− σ√*T*3 ))

0

*φ*= *T* (*r*− σ2)

2 6

*Z* = *X* +*c*(*Y* −μ γ) (19)

*j j j*

Lastly, the option price using the control variate is calculated simply as

*C* = 1 ∑*N Z v N* =1 *j*

*j*

To illustrate the efficiency of the control variate method, a test was conducted where the prices of an Asian fixed strike call option were calculated using both the arithmetic average, as well as the control variate method. Monte Carlo simulations with 30 steps in each simulation were conducted with an increasing amount of simulations to see at which point the price stabilizes. The results can be seen in table 1, where N is the amount of simulations run.

*Table 1: Comparison between vanilla arithmetic call option and control variate call option*

M![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.012.png)

*Figure 4: Price estimations of vanilla arithmetic call option and control variate call option*

![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.013.png)

By looking at figure 4 and table 1, it is evident that the control variate method is highly efficient. The standard deviation is instantly almost 99% smaller, and the price is stable at around 1.5 almost instantly. Hence, when performing the deep hedging, the control variate method will be used for estimating the price of an arithmetic Asian fixed strike call option.

4. **Delta hedging**

Traditionally traders use “Greek letters” to hedge the risks associated with derivatives. Delta (*N(d1))*, which is the partial derivative of the value of a transaction with respect to the underlying asset price, is the most often used of the Greek letters. Usually traders try to maintain a delta-neutral position. The delta of the option changes as time progresses and thus the position must be rebalanced at certain intervals. In optimal conditions the position would be rebalanced continuously, which could lead to the cost of hedging the option equal its theoretical price. In practice, transaction costs and other trading costs, as well as liquidity constraints imply that the strategy should be modified. In addition to these trading costs, realistically the positions are rebalanced periodically, instead of being balanced continuously (Cao, Hull, Zissis & Poulos 2019). 

The delta of an option is the sensitivity of the option price to a change in the price of the underlying asset. Using the Black Scholes model option pricing formula C, delta can be expressed as 

ln( *S* )+(*r*+σ2/2)*T* Δ= ∂*C*= *N* ( *K*~~ ) ![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.014.png)∂*S* σ√*T*

The fact that there are no closed form equations for arithmetic Asian options, and technically no delta for them, makes Asian options particularly interesting subjects for deep hedging, where the hedging costs are attempted to be minimized by other factors the Greeks. 

5. **Neural networks**

Feedforward  neural  networks  are the  quintessential  deep learning models. The goal  of  a feedforward network is to approximate a function *f\**. For instance, for a classifier, *y = f\*(**x**)* maps an input ***x*** to a category *y*. A feedforward neural network (*Neural network*) defines a mapping ***y** = f(**x;θ**)* and learn the value of the parameters ***θ*** that result in the best function approximation. These neural network models are called feedforward as there are no feedback connections in which the model outputs are fed back into itself (Goodfellow et al. 2016). 

![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.015.png)

***Figure 5: An example of a feedforward neural networks structure***

Neural networks are called networks because they are typically a combination of multiple different functions. In figure two, there are four different functions, *f(1),, f(2), f(3), f(4)* which are respectively the input layer  ℝ6 , the two hidden layers  ℝ12 and the output layer  ℝ1 . These are connected to

*i*

a chain to form  *f* (*x*)= *f* (4)(*f* (3)(*f* (2)(*f* (1)(*x*)))) . 

Each individual data point, apart from the input data, is called a node or a neuron in the network. A neuron is a computational unit that has m amounts of weighted input connections, and can be seen as a linear regression model as follows

*y*=∑*m w x* +*b*

*i i*

*i*=1

where *b* is the bias, w is the weight and x is the input data. 

The most appealing property of neural networks is the ability to adapt their behaviour as the characteristics of the system changes. In order for neural networks to be more accurate, the input of a node is run through an activation function before the neuron gets its final value. Without an activation function, the output signal would merely be a simple linear function which is just a polynomial degree of one. The most commonly used activation functions are non-linear functions, which are used to induce non-linearities into the network (Sharma & Sharma 2020). 

*Figure 6: An illustration of a single node in the network*

![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.016.png)

Lastly, a feedforward neural network can be defined as follows: 

*Let L, N0, N1,…, N* ∈ℕ *with L≥2, let*  σ:ℝ ⇒ ℝ *and for any l = 1,…,L, let  W* :ℝ*N l*−1⇒ ℝ *Nl an*

*L l*

*affine function. A function  F*:ℝ *N*0⇒ ℝ *NL defined as*

*F*(*x*)=*W* ⋅*F* ⋅...⋅*F with  F* =σ⋅*W for  l*=1*,*...*,L*−1

*L L*−1 1 *l l*

In the definition above, σ is the activation function, *L* denotes the number of layers, *N* ,...*N* denote

*1 L-1*

the dimensions of the hidden layers and *N* , *N* are the input and output layers, respectively (Buehler

*0 L*

et al. 2019).  

**2.5.1 Recurrent Neural Networks**

One shortcoming in feedforward neural networks is that are unable to take the past into consideration. Oftentimes however, it would be useful for a model to be able to take the past in consideration. Recurrent neural networks also include feedback connections, which enable them to use past data (Goodfellow, Bengio & Courville 2016).* 

*Figure 7: Illustration of a traditional architecture in a recurrent neural network*

![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.017.png)

The hidden layer activation *a* and the output *y* at time t can* be expressed as follows

*a* =ϕ(*W a* +*W X* +*b* )

*t aa t*−1 *ax t a*

where *W*  terms denote weight  matrices which connect  the inputs to  the hidden  layer *a* correspondingly, *b*  denotes bias vectors and *ɸ* is the hidden layer function. Given the hidden sequences, the output of the sequence can be expressed as

*m*

^*y*=*b* +∑ *W hi y hi y t*

*i*=1

*y* = γ(*y*^)

*t t*

where *Ɣ* is the output layer function. Hence, the complete network defines a function, parametrised by the weight matrices from input histories x to the output vectors y (Graves 2013).

1:t t  

6. **Deep hedging**

In the absence of trading constraints, such as trading costs, the Black-Scholes model can yield and maintain a perfect hedge. It is evident that such an assumption does not reflect real markets. There have been several different strategies that attempt improving the Black-Scholes model under the presence of transaction costs, such as in the study of Leland (1985), where the researcher extended the Black-Scholes model by replicating a portfolio to a discretely rebalanced portfolio. Deep hedging (Buehler et al. 2019) is a ground-breaking framework, which especially in the presence of trading costs and other constraints has been shown to often outperform the traditional hedging strategies, thus drastically reducing the cost of hedging. Deep hedging represents a hedging strategy using deep learning models. In the research paper of Buehler et al. the authors used a deep semi- recurrent neural network model and demonstrated that the deep hedging model could be feasibly used to hedge a European option under exponential utility. 

**3 Deep heding for an Asian fixed strike call option**

Following the steps of Buehler et al. 2019, a neural network model is constructed for essentially creating a pseudo-delta, which the neural network model creates and that is not related to the traditional Greeks. Additionally, the model attempts to minimize the cost of hedging. 

The chosen network structure used in the research paper is inspired by the study of Beuhler et al. 2019. In order to calculate each pseudo-delta at a given step in a simulation, the following network architecture was built:

`  `*Table 2: Deep hedging neural network architecture*



|Input|*input* |
| - | - |
|Layer 1|*Dense(32, activation = ‘relu)*|
|Layer 2|*Dropout(0.5)*|
|Layer3|*Dense(32, activation = ‘relu’)*|
|Layer 4|*Dense(32, activation = ‘leaky\_relu’)*|
|Output|*Dense(1)*|
The model used Adam optimizer and a mean-squared-error loss function. After the pseudo-delta for each step in the simulation is calculated, the model tries to maintain a delta-neutral position, and ultimately minimize the cost of hedging. This can be described as:

[∑*n* Δ (*S* −*S* )]+*max*(*A* − *K,*0)≈*C*  (20)

*i i i*+1 *T*

*i*=0

where *Δ* is the pseudo-delta estimated by the neural network model, *At* is the arithmetic average of the price sequence *S*, and *C* is the price of an Asian fixed strike call option calculated with the control variate method (*see equation 19*). For the scenarios tested out, S is 100, time to maturity is

0

30/365, and the risk-free interest rate is 0. Each simulation that is run consists of 30 steps.

**3.1 Results**

The performance of the model was tried out in different conditions. The validation results from different scenarios are displayed below.

*Figure 8: Validation results with different strike prices*

14

![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.018.png)

1) *K = 97* 

![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.019.png)

2) *K = 100*

![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.020.png)

3) *K = 103*



*Figure 9: Validation results with different volatilities*

![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.021.png) ![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.019.png) ![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.022.png)

*a) σ = 0.1      a) σ = 0.2             a) σ = 0.4*

From the results above, it can be seen that the adapts well into different scenarios. The most challenging situations for the model seem to be higher volatility situations, which is natural, since that’s the expectation for all hedging models. However, it is noteworthy that the hedging results could probably be made significantly better. By further analysing the model performance, it can be seen from figure 8 that the mean-squared-error of the model is around 0.27, regardless of the amount of epochs. This could be due to slight inaccuracies in the Monte Carlo pricing method, even when using the control variate.  

*Figure 8: MSE loss function at different epochs*

![](Aspose.Words.4fbcf997-1d6f-4183-a364-1e521e59a0f9.023.png)

**4 Conclusions**

This research paper discussed deep hedging methods with a focus on Asian fixed strike options. Different pricing methods for Asian options were introduced and analysed, whereafter the fundamental concepts of neural networks were reviewed. The empirical part of the paper aimed to hedge Asian options by creating a pseudo-delta and trying to minimize the profit and loss function for hedging. The neural networks model does a good job at estimating the pseudo-delta, but further improvements could be made to increase the accuracy of the model. Perhaps a binomial option pricing method done in a similar fashion as in the research paper of Hull & White (1993), would lead to a more accurate option price, thus improving the accuracy of the model (*recall chart 20*). Alternatively, more focus could be laid on optimizing the neural network model by hyperparameter tuning and building pipelines for creating various models. Recurrent neural networks could be an interesting alternative, as they take the past actions into consideration. 

**References**

P.  Glasserman,  Monte  carlo  methods  in  financial  engineering,  Applications  of mathematics : stochastic modelling and applied probability, Springer, 2004.

Goodfellow, I., Bengio, Y. and Courville, A., Deep Learning, 2016 (MIT Press). Available online at: http://www.deeplearningbook. Org.

Lakhlani, V. B. (2013). Pricing and Hedging Asian Options.

Zhang, H. (2009). *Pricing Asian Options using Monte Carlo Methods*.

Joy, C., Boyle, P. P., & Tan, K. S. (1996). Quasi-Monte Carlo methods in numerical finance. *Management science*, *42*(6), 926-938.

Galda, G. (2008). Variance Reduction for Asian Options.

Cox, J. C., Ross, S. A., Rubinstein, M., “Option Pricing: A Simplified Approach”, Journal of Financial Economics (1979)

Vajjha, K. (2017). Module 3 Exam Solutions.

Hull, J. C., & White, A. D. (1993). Efficient procedures for valuing European and American path-dependent options. *The Journal of Derivatives*, *1*(1), 21-31.

Black, F., & Scholes, M. (2019). The pricing of options and corporate liabilities. In *World Scientific Reference on Contingent Claims Analysis in Corporate Finance: Volume 1: Foundations of CCA and Equity Valuation* (pp. 3-21).

Rogers, L. C. G., & Shi, Z. (1995). The value of an Asian option. *Journal of Applied Probability*, *32*(4), 1077-1088.

Kemna, A. G., & Vorst, A. C. (1990). A pricing method for options based on average asset values. *Journal of Banking & Finance*, *14*(1), 113-129.

Boyle, P. P. (1977). Options: A monte carlo approach. *Journal of financial economics*, *4*(3), 323-338.

Cao, J., Chen, J., Hull, J., & Poulos, Z. (2021). Deep hedging of derivatives using reinforcement learning. *The Journal of Financial Data Science*, *3*(1), 10-27.

Sharma, S., Sharma, S., & Athaiya, A. (2017). Activation functions in neural networks. *towards data science*, *6*(12), 310-316.

Graves, A. (2013). Generating sequences with recurrent neural networks. *arXiv preprint arXiv:1308.0850*.

Leland, H. E. (1985). Option pricing and replication with transactions costs. *The journal of finance*, *40*(5), 1283-1301.
17

[^1]: *Y* =*e*−*rT*[*max*((∏*N S* )*N*1)− *K,*0]
[^2]: ` `*i*

    *i*=0
