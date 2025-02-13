#%% 
# poll simulation:
import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

with st.sidebar:
    # population_size = st.number_input('Total Population', value=10_000_000)
    st.write('Parameters')
    P = st.number_input(r'True Percentage of 1 Among Population, $P_{true}$ (%)', value=55.0) / 100
    max_sample_size = st.number_input('Max Sample Size', value=1000)
    conf = st.number_input(r'Confidence Level, $CL$ (%)', value=95.0) / 100
    seed = st.number_input('Random Seed', value=0)

# generate population
np.random.seed(seed)

# generate sample
sample_size = np.arange(10, max_sample_size, 10, dtype=int)
# Initialize the list of lists
sample = []
# Generate the list of lists
for i, size in enumerate(sample_size):
    if i == 0:
        # For the first element, generate a new list of random values
        sample.append(np.random.choice([1, 0], size=size, p=[P, 1-P]).tolist())
    else:
        # Reuse the previous list and append additional random values
        prev_list = sample[i - 1]
        additional_values = np.random.choice([1, 0], size=size - len(prev_list), p=[P, 1-P]).tolist()
        sample.append(prev_list + additional_values)

# standard error
x_bar = np.array([np.mean(sample, axis=0) for sample in sample])
SE = np.array([np.sqrt(x_bar * ( 1 - x_bar ) / sample_size) for x_bar, sample_size in zip(x_bar, sample_size)])

# confidence level
conf_array = np.arange(conf - 0.02, conf + 0.02, 0.01)
conf_array = conf_array[conf_array<1]
prob_within_interval = [np.array([norm.cdf((1-conf) / SE) - norm.cdf(-(1-conf) / SE) for SE in SE]) for conf in conf_array]

# calculate p-value for each sample size (Null hypothesis: P = 0.5)
p_value = np.array([2 * (1 - norm.cdf(np.abs(0.5 - x_bar) / SE)) for x_bar,SE in zip(x_bar,SE)])

# make a subplot with secondary y axis
fig_mean = go.Figure()
fig_mean.add_trace(go.Scatter(
    x=sample_size, 
    y=x_bar + 1.96*SE, 
    mode='lines', 
    name='Sample mean + SE', 
    line=dict(color='lightblue'),
    showlegend=False,
))
fig_mean.add_trace(go.Scatter(
    x=sample_size, 
    y=x_bar, 
    mode='lines', 
    name='Sample mean',
    line=dict(color='blue'),
    fill='tonextx',
    fillcolor='lightblue'
))
fig_mean.add_trace(go.Scatter(
    x=sample_size, 
    y=x_bar - 1.96*SE, 
    mode='lines', 
    name='Sample mean - SE', 
    line=dict(color='lightblue'),
    showlegend=False,
    fill = 'tonextx',
    fillcolor='lightblue'
))
fig_mean.add_trace(go.Scatter(
    x=sample_size, 
    y=[P]*len(sample_size), 
    mode='lines', 
    name='True Percentage',
    line=dict(color='red', dash = 'dash'),
))
fig_mean.update_layout(
    title = 'Poll Simulation, Sample Mean',
    xaxis_title = 'Sample Size',
    yaxis_title = 'Sample Mean',
    # legend on the graph
    legend=dict(
        yanchor="top",
        y=1.1,
        xanchor="left",
        x=0,
        # horizontal arrangement
        orientation="h"
    ),
)

fig_prob = go.Figure()
for i,con in enumerate(conf_array):
    fig_prob.add_trace(go.Scatter(
        x=sample_size, 
        y=prob_within_interval[i], 
        mode='lines', 
        name=f'CL = {con:.0%}',
    ))
fig_prob.update_layout(
    title = 'Poll Simulation, Confidence Level',
    xaxis_title = 'Sample Size',
    yaxis_title = 'Chance of Being Within CL',
)

fig_p_value = go.Figure()
fig_p_value.add_trace(go.Scatter(
    x=sample_size, 
    y=p_value, 
    mode='lines', 
    name='P-value',
    line=dict(color='purple'),
))
fig_p_value.add_trace(go.Scatter(
    x=sample_size, 
    y=[0.05]*len(sample_size), 
    mode='lines', 
    name='Significance Level',
    line=dict(color='red', dash = 'dash'),
))

fig_p_value.update_layout(
    title = 'Poll simulation - p-value',
    xaxis_title = 'Sample Size',
    yaxis_title = 'p-value',
    # legend on the graph
    legend=dict(
        yanchor="top",
        y=1.1,
        xanchor="left",
        x=0,
        orientation="h"
    ),
)

st.title('Poll Simulation')
st.write(
    """
    This dashboard simulates a poll with a given population and true percentage of 1 among the population.
    The pollster takes samples of different sizes and calculates different statistics.
    """)
st.write(
    """
    The first graph shows the sample mean and the confidence interval around the sample mean.
    The sample mean is simply calculated as:

    $$ \\bar{X} = \\frac{1}{n} \\sum_{i=1}^{n} x_i $$
    
    The confidence interval is calculated as:
    
    $$ CI = \\bar{X} \pm 1.96 \\times SE $$
    
    where $SE$ is the standard error within the sample of size $$n$$, calculated as:
    $$ SE = \\sqrt{\\frac{\\bar{X} \\times (1 - \\bar{X})}{n}} $$
    """)
st.plotly_chart(fig_mean)

st.write(
    """
    The second graph shows how confident the pollster is that the sample mean is within a given confidence level ($$CL$$) of the true percentage.
    the graph is constructed for different confidence levels.
    This probability can be expressed as follows:

    $$P(\\left|\\bar{X} - P_{true}\\right| < 1-CL)$$

    This is the probability that the difference between the sample mean and the true percentage is less than $$1-CL$$, where $$CL$$ is confidence level.
    This probability can be simplified as follows:

    $$ P(\\bar{X} \\le P_{true} + (1-CL)) - P(\\bar{X} \\le P_{true} - (1-CL)) $$

    At this point, we use this approximation for standard error ($$SE$$):

    $$ SE(\\bar{X}) = \\sqrt{\\frac{P_{true} \\times (1 - P_{true})}{n}} \\approx \\sqrt{\\frac{\\bar{X} \\times (1 - \\bar{X})}{n}}$$

    The probability in question can be further simplified to:

    $$ P((1-CL) / SE(\\bar{X})) - P(-(1-CL) / SE(\\bar{X}))$$
    """)
st.plotly_chart(fig_prob)

st.write(
    """
    The third graph shows the p-value for each sample size. In this case, the null hypothesis is that the true percentage is 0.5, ($$P_{true} = 0.5$$).
    The p-value is calculated as follows:

    $$ \\text{p-value} = 2 \\times \\left(1 - \\Phi\left(\\left|\\frac{\\bar{X} - 0.5}{SE(\\bar{X})}\\right|\\right)\\right) $$

    where $$\\Phi$$ is the cumulative distribution function of the standard normal distribution.

    The significance level is normally set to 0.05. Interpretation of this graph is simple: if the p-value is less than 0.05, we reject the null hypothesis.
    Meaning to say, we can say that the true percentage is not 0.5 and one party is more popular than the other.
    """)
st.plotly_chart(fig_p_value)

