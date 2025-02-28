#%% 
# poll simulation:
import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# wide layout
# st.set_page_config(layout="wide")

with st.sidebar:
    # population_size = st.number_input('Total Population', value=10_000_000)
    st.write('Parameters')
    P_true = st.number_input(r'True Percentage of 1 Among Population, $P_{\text{true}}$ (%)', value=55.0) / 100
    max_sample_size = st.number_input('Max Sample Size', value=1000)
    conf = st.number_input(r'Confidence Level, $CL$ (%)', value=95.0) / 100
    seed = st.number_input('Random Seed', value=42)
    st.write('Pollster Bias')
    num_pollsters = st.slider('Number of Pollsters', min_value=1, max_value=20, value=5)
    max_bias = st.slider(r'Maximum Bias, $B_\text{max}$ (%)', min_value=0.0, max_value=10.0, value=2.0) / 100

# [10, 20, 30, ..., max_sample_size]
samples_arr = np.arange(10, max_sample_size, 10, dtype=int)

def create_sample(P):
    # generate population
    np.random.seed(seed)
    
    # Initialize the list of lists
    sample = []
    # Generate the list of lists
    for i, size in enumerate(samples_arr):
        if i == 0:
            # For the first element, generate a new list of random values
            sample.append(np.random.choice([1, 0], size=size, p=[P, 1-P]).tolist())
        else:
            # Reuse the previous list and append additional random values
            prev_list = sample[i - 1]
            additional_values = np.random.choice([1, 0], size=size - len(prev_list), p=[P, 1-P]).tolist()
            sample.append(prev_list + additional_values)
    return sample

# standard error
def calculate_sample_statistics(max_sample_size, sample):
    sample_size = np.arange(10, max_sample_size, 10, dtype=int)

    # calculate sample mean and standard error
    x_bar = np.array([np.mean(sample, axis=0) for sample in sample])
    SE = np.array([np.sqrt(x_bar * ( 1 - x_bar ) / sample_size) for x_bar, sample_size in zip(x_bar, sample_size)])
    
    # confidence level
    conf_array = np.arange(conf - 0.02, conf + 0.02, 0.01)
    conf_array = conf_array[conf_array<1]
    prob_within_interval = [np.array([norm.cdf((1-conf) / SE) - norm.cdf(-(1-conf) / SE) for SE in SE]) for conf in conf_array]
    
    return x_bar, SE, prob_within_interval, conf_array

sample = create_sample(P_true)
x_bar, SE, prob_within_interval, conf_array = calculate_sample_statistics(max_sample_size, sample) 

# calculate p-value for each sample size (Null hypothesis: P = 0.5)
p_value = np.array([2 * (1 - norm.cdf(np.abs(0.5 - x_bar) / SE)) for x_bar,SE in zip(x_bar,SE)])

# make a subplot with secondary y axis
fig_mean = go.Figure()
fig_mean.add_trace(go.Scatter(
    x=samples_arr, 
    y=x_bar + 1.96*SE, 
    mode='lines', 
    name='Sample mean + SE', 
    line=dict(color='lightblue'),
    showlegend=False,
))
fig_mean.add_trace(go.Scatter(
    x=samples_arr, 
    y=x_bar, 
    mode='lines', 
    name='Sample mean',
    line=dict(color='blue'),
    fill='tonextx',
    fillcolor='lightblue'
))
fig_mean.add_trace(go.Scatter(
    x=samples_arr, 
    y=x_bar - 1.96*SE, 
    mode='lines', 
    name='Sample mean - SE', 
    line=dict(color='lightblue'),
    showlegend=False,
    fill = 'tonextx',
    fillcolor='lightblue'
))
fig_mean.add_trace(go.Scatter(
    x=samples_arr, 
    y=[P_true]*len(samples_arr), 
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
conf_array = np.arange(conf - 0.02, conf + 0.02, 0.01)
conf_array = conf_array[conf_array<1]

prob_within_interval = [np.array([norm.cdf((1-conf) / SE) - norm.cdf(-(1-conf) / SE) for SE in SE]) for conf in conf_array]

for i,con in enumerate(conf_array):
    fig_prob.add_trace(go.Scatter(
        x=samples_arr, 
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
    x=samples_arr, 
    y=p_value, 
    mode='lines', 
    name='P-value',
    line=dict(color='purple'),
))
fig_p_value.add_trace(go.Scatter(
    x=samples_arr, 
    y=[0.05]*len(samples_arr), 
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
    $$ SE = \\sqrt{\\frac{1}{n}\\bar{X} \\times (1 - \\bar{X})} $$
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

    $$ P(\\bar{X} \\le P_{\\text{true}} + (1-CL)) - P(\\bar{X} \\le P_{\\text{true}} - (1-CL)) $$

    At this point, we use this approximation for standard error ($$SE$$):

    $$ SE(\\bar{X}) = \\sqrt{\\frac{1}{n}P_{\\text{true}} \\times (1 - P_{\\text{true}})} \\approx \\sqrt{\\frac{1}{n}\\bar{X} \\times (1 - \\bar{X})}$$

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

# biased pollsters
st.header('Biased Pollsters')
st.write("""
In this section, we will discuss how biased pollsters can affect the results of the poll.
The pollster can be biased. In this case, the pollster can be biased towards one party. 
The bias can be expressed as a percentage. 

Each pollster will have:
         
* A bias ($B$): This represents how much their reported results deviate from the true population percentage. 
         For example, a bias of +0.05 means the pollster overestimates the true percentage by 5%.

Everytime a pollster draws a sample, the pollster will draw a sample from the population with a bias:

$$ P(x = 1) = P_{\\text{true}} + B $$

We use Monte Carlo simulation to simulate the pollsters.
A maximum bias is set for the pollsters $B_\\text{max}$. 
The pollsters will have a bias between $-B_\\text{max}$ and $+B_\\text{max}$ distributed uniformly.
Also, the sample size of the pollsters is randomly selected between 0.5 and 1.5 times the max sample size parameter set by the user in the sidebar.

The aggregated results of all pollsters are calculated as follows:
         
$$ \\bar{X}_{\\text{agg}} = \\frac{\\sum_{i=1}^{N} \\bar{X}_i \\times w_i}{\\sum_{i=1}^{N} w_i} $$
         
$$ SE_{\\text{agg}} = \\sqrt{\\frac{1}{\\sum_{i=1}^{N} w_i}} $$
         
where $N$ is the number of pollsters, $\\bar{X}_i$ is the sample mean of the $i$th pollster, $w_i$ is the weight of the $i$th pollster, and $SE_i$ is the standard error of the $i$th pollster.
Weight of the pollster is calculated as the inverse of the standard error of the pollster:
         
$$ w_i = \\frac{1}{SE_i^2} $$
         
Here, note that the weight of each pollster is indirectly proportional to the sample size of the pollster, meaning that the larger the sample size, the smaller the standard error, and the larger the weight.
         
$ w_i \propto N_i $
         
""")

# Generate synthetic pollster data
pollster_results = [None] * num_pollsters
for i in range(num_pollsters):
    sample_size = np.random.randint(max_sample_size*0.5, max_sample_size*1.5)
    bias = np.random.uniform(-max_bias, max_bias)
    sample_pollster = np.random.choice([1, 0], p=[P_true + bias, 1 - P_true - bias], size=sample_size)
    x_bar = np.mean(sample_pollster)
    SE_pollster = np.sqrt(x_bar * (1 - x_bar) / sample_size)

    pollster_results[i] = {
        'Sample Size': sample_size,
        'Bias': bias,
        'Sample Mean': x_bar,
        'Standard Error': SE_pollster
    }


# Convert to DataFrame for easier manipulation
import pandas as pd

df = pd.DataFrame(pollster_results)
df.index.name = 'Pollster'
df = df.reset_index()

# Aggregate results of all pollsters
# Weighted average of sample means
weights = 1 / df['Standard Error']**2
x_bar_agg = np.sum(df['Sample Mean'] * weights) / np.sum(weights)

# Combined standard error
SE_agg = np.sqrt(1 / np.sum(weights))

# Display aggregated results
# st.write(f"Aggregated Sample Mean: {x_bar_agg:.4f}")
# st.write(f"Aggregated Standard Error: {SE_agg:.4f}")

# create a plot of all pollsters average value and their confidence intervals
# each pollster is a horizontal line with confidence interval
# the x axis is the percentage of 1 in the population
# the y axis is the number of pollsters
fig_bias = make_subplots(
    rows=1, cols=3, 
    shared_yaxes=True,
    # gap between the plots
    column_widths=[0.15, 0.7, 0.15],
    subplot_titles=['Bias Values', 'Pollster Results', 'Sample Size'],
    horizontal_spacing=0.05,
)

fig_bias.add_trace(go.Bar(
    x=df['Bias'],
    y=df['Pollster'],
    orientation='h',
    name='Bias',
    marker=dict(color='blue'),
    showlegend=False,
), row=1, col=1)

# add the 0 biasline
fig_bias.add_vline(
    x=0,
    line=dict(color='grey', dash='dash'),
    row=1, col=1
)


for i in range(num_pollsters):
    fig_bias.add_trace(go.Scatter(
        x=[df['Sample Mean'][i] - 1.96 * df['Standard Error'][i], df['Sample Mean'][i] + 1.96 * df['Standard Error'][i]],
        y=[i, i],
        mode='lines',
        name=f'Pollster {i}',
        line=dict(color='blue', width=5),
        showlegend=False,
    ), row=1, col=2)
    fig_bias.add_trace(go.Scatter(
        x=[df['Sample Mean'][i]],
        y=[i],
        mode='markers',
        name=f'Pollster {i}',
        marker=dict(color='blue', size=10),
        showlegend=False,
    ), row=1, col=2)

fig_bias.add_vline(
    x=P_true,
    line=dict(color='red', dash='dash'),
    annotation_text='P true',
    annotation_position='bottom right',
    annotation_font_color='red',
    row=1, col=2
)

# add the 50% line
fig_bias.add_vline(
    x=0.5,
    line=dict(color='grey', dash='dash'),
    annotation_text='50%',
    annotation_position='bottom right',
    annotation_font_color='grey',
    row=1, col=2
)

# add the aggregated value
fig_bias.add_trace(go.Scatter(
    x=[x_bar_agg - 1.96 * SE_agg, x_bar_agg + 1.96 * SE_agg],
    y = [num_pollsters, num_pollsters],
    mode='lines',
    name='Aggregated Value',
    line=dict(color='green'),
    showlegend=True,
), row=1, col=2)

fig_bias.add_trace(go.Scatter(
    x=[x_bar_agg],
    y=[num_pollsters],
    mode='markers',
    name='Aggregated Value',
    marker=dict(color='green', size=10),
    showlegend=False,
), row=1, col=2)

# show a bar chart of number of samples in each pollster on the right
fig_bias.add_trace(go.Bar(
    x=df['Sample Size'],
    y=df['Pollster'],
    orientation='h',
    name='Sample Size',
    marker=dict(color='blue'),
    showlegend=False,
), row=1, col=3)

# show average sample size
fig_bias.add_vline(
    x=max_sample_size,
    line=dict(color='red', dash='dash'),
    annotation_text='N_avg',
    row=1, col=3
)

fig_bias.update_layout(
    xaxis_title = 'Bias',
    xaxis2_title='True Percentage',
    yaxis_title='Pollster',
    xaxis3_title='Sample Size',
    xaxis_tickformat = '.0%',
    xaxis2_tickformat = '.0%',
    # legend on top
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.2,
        xanchor='right',
        x=0.2
    ),
)

st.plotly_chart(fig_bias)
