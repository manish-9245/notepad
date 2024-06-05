import pandas as pd
import numpy as np
import random
from datetime import datetime

def calculate_principal_amt(qty, factor, price):
    return qty * (factor / 100) * (price / 100)

def calculate_full_commission(price, qty, comm_amt, comm_type):
    if comm_type == 'R':
        return (price * qty) + (comm_amt * qty)
    elif comm_type == 'F':
        return (price * qty) + comm_amt
    else:
        return price * qty

def calculate_indicator(ind):
    if ind == 'B':
        return "BUY"
    elif ind == 'S':
        return "SELL"
    elif ind == 'SS':
        return "Short Sell"
    else:
        return "Unknown"

input_cols = ['commission_type', 'price', 'quantity', 'commission_amount', 'factor', 'indicator_short']
output_cols = ['principal_amt', 'full_commission', 'indicator_long']

comm_type = ['R', 'F', 'B']
ind_short = ['B', 'S', 'SS']

input_dict = dict()
output_dict = dict()

rows = 5000
for i in range(rows):
    inp_com_type = random.choice(comm_type)
    inp_price = random.uniform(0, 1000000)
    qty = random.randint(0, 1000000)
    inp_comm_amt = random.uniform(0, 1000000)
    factor = random.random()
    inp_ind = random.choice(ind_short)

    input_dict[i] = dict()
    input_dict[i]["commission_type"] = inp_com_type
    input_dict[i]["price"] = inp_price
    input_dict[i]["quantity"] = qty
    input_dict[i]["commission_amount"] = inp_comm_amt
    input_dict[i]["factor"] = factor
    input_dict[i]["indicator_short"] = inp_ind

    principal = calculate_principal_amt(qty, factor, inp_price)
    full_commission = calculate_full_commission(inp_price, qty, inp_comm_amt, inp_com_type)
    indicator_long = calculate_indicator(inp_ind)

    output_dict[i] = dict()
    output_dict[i]['principal_amt'] = principal
    output_dict[i]['full_commission'] = full_commission
    output_dict[i]['indicator_long'] = indicator_long

inp_df = pd.DataFrame.from_dict(input_dict, orient='index')
inp_df.to_csv('input_{}.csv'.format(datetime.today().strftime('%Y-%m-%d')), index=False)

out_df = pd.DataFrame.from_dict(output_dict, orient='index')
out_df.to_csv('output_{}.csv'.format(datetime.today().strftime('%Y-%m-%d')), index=False)