

"""
Main class definition for running Market Match and Causal Impact analysis.
"""

import numpy as np
import pandas as pd

import dtw as dtw
from causalimpact import CausalImpact


class R_MarketMatching():
    
    """
    TODO:

    add the functions: check_inputs() ; cmean() ; range01()

    replace np.arange() with df.cumcount()+1

    """


    def read_data(filename):

        """
        read input weather data and convert string date column to date type

        """

        ip_df = pd.read_csv(filename)

        ip_df['Date'] = pd.to_datetime(ip_df['Date'])

        return ip_df


    def sapply(x):
        """
        function for log_plus()
        """
        if x <= 0:
            op = 0.00
        else:
            op = np.log(x)

        return op


    def log_plus(x):

        return list(map(sapply(), x)) #clean up: correct way to pass sapply to map?

    @staticmethod
    def create_market_vectors(ip_df, test_market, ref_market):

        d = ip_df[ip_df['match_var'].notna()]
        test = d[d['id_var'] == test_market]
        test = test[['date_var', 'match_var']]
        col_map = {'match_var': 'y'}
        test = test.rename(columns = col_map)

        # ref_market should be a list
        if not type(ref_market) == list:
            ref_market = [ref_market]

        if len([ref_market]) == 1:
            ref = d[d['id_var'] == ref_market[0]]
            ref = ref[['date_var', 'match_var']]
            col_map = {'match_var': 'x1'}
            ref = ref.rename(columns = col_map)

            f = test.merge(ref, on='date_var', how='inner')

        else:

            #TODO check this block from line 169 in R
            d = d.drop_duplicates()
            ref = d[d['id_var'].isin(ref_market)]
            ref = ref['date_var', 'id_var', 'match_var']

            ref = pd.pivot_table(ref, values = 'match_var', index=['date_var'], columns='id_var').reset_index()

            x_var_list = ['date_var'] + ['x%d' % i for i in range(len(ref_market))]

            ref.columns = x_var_list

            f = test.merge(ref, on='date_var', how='inner')
            f.dropna(inplace=True)

        return f


    def mape_no_zeros(test, ref):
        """
        not used so far - will be need after analyze_betas is added
        """
        df = pd.concat(test, ref, axis=1)
        df = df[(df['test'] > 0) & (df['ref'] > 0)]

        return np.mean(np.abs(df['test'] - df['ref'])/df['test'])


    def lagp(x, p):
        """
        this function is only called by the function dw()
        """

        op = [0]*p + x[:(len(x) - p)]

        return op


    def dw(y, yhat):
        """
        this function has not been used in the code yet
        will be needed when analyze_betas is added
        """
        res = y - yhat
        lag_res = lagp(res, 1)

        r = np.corrcoef(res[1:len(res)], lag_res[1:len(lag_res)])

        return 2*(1-r)

    @staticmethod
    def calculate_distances(markets_to_be_matched, ip_df, id_variable, i, warping_limit,
                            matches, dtw_emphasis):

        """
        core function that uses DTW to compute distances between test and reference time series
        """

        if dtw_emphasis == 0:
            warping_limit = 1

        row = 0
        this_market = markets_to_be_matched[i]
        shape_ini = (len(ip_df['id_var']), 9)
        emp_arr = np.empty(shape_ini)
        distances_df = pd.DataFrame(data=emp_arr)
        col_headers = [id_variable, "BestControl", "RelativeDistance", "Correlation",
                        "Length", "SUMTEST", "SUMCNTL", "RAWDIST", "Correlation_of_logs"] #clean up - add "Skip"??
        distances_df.columns = col_headers

        distances_df[id_variable] = distances_df[id_variable].astype('str')
        distances_df['BestControl'] = distances_df['BestControl'].astype('str')


        messages = 0

        # for each market
        for j in range(ip_df['id_var'].nunique()):
            is_valid_test = True
            unq_mkts = list(ip_df['id_var'].unique())
            that_market = unq_mkts[j]
            distances_df.at[row, id_variable] = this_market
            distances_df.at[row, 'BestControl'] = that_market
            mkts = R_MarketMatching.create_market_vectors(ip_df, this_market, that_market)
            test = mkts[['y']]
            ref = mkts[['x1']]
            # dates = mkts[['date_var']] this is in the R code, but not used anywhere
            sum_test = None
            sum_cntl = None
            dist = 0

            # If insufficient data or no variance
            if (np.isclose(np.var(test), 0, atol=0.001) or
                len(test) <= 2*warping_limit+1 or np.sum(np.abs(test)).iloc[0] == 0):
                is_valid_test = False
                messages += 1

            # If data and variance are sufficient and test vector was valid
            if (this_market != that_market and is_valid_test == True and np.var(ref)[0] > 0 and len(test) > 2*warping_limit):
                sum_test = np.abs(np.sum(test)).iloc[0]
                sum_cntl = np.abs(np.sum(ref)).iloc[0]
                if (dtw_emphasis > 0 and sum_test > 0):
                    dtw_op = dtw.dtw(test, ref, window_type='sakoechiba',
                                        window_args={'window_size': warping_limit})
                    raw_dist = dtw_op.distance
                    dist = raw_dist/sum_test

                else:
                    dist = 0
                    raw_dist = 0

                distances_df.at[row, 'Correlation'] = test['y'].corr(ref['x1'])
                distances_df.at[row, 'RelativeDistance'] = dist
                # this is the first time that column "Skip" is introduced in the code
                distances_df.at[row, 'Skip'] = False
                distances_df.at[row, 'Length'] = len(test)
                distances_df.at[row, 'SUMTEST'] = sum_test
                distances_df.at[row, 'SUMCNTL'] = sum_cntl
                distances_df.at[row, 'RAWDIST'] = raw_dist

                if (np.max(ref).iloc[0] > 0 and np.max(test).iloc[0] > 0):
                    distances_df.at[row, 'Correlation_of_logs'] = test['y'].corr(ref['x1'])

                else:
                    distances_df.at[row, 'Correlation_of_logs'] = None

            else:
                if this_market != that_market:
                    messages += 1

                distances_df.at[row, 'Skip'] = True
                distances_df.at[row, 'RelativeDistance'] = None
                distances_df.at[row, 'Correlation'] = None
                distances_df.at[row, 'Length'] = None
                distances_df.at[row, 'SUMTEST'] = None
                distances_df.at[row, 'SUMCNTL'] = None
                distances_df.at[row, 'RAWDIST'] = None
                distances_df.at[row, 'Correlation_of_logs'] = None

            row += 1

        if messages > 0:
            print(f'{messages} markets were not matched with {this_market} \
                due to insufficient data or no variance.')

        distances_df['matches'] = matches
        distances_df['w'] = dtw_emphasis
        distances_df['matching_start_date'] = np.min(ip_df['date_var'])
        distances_df['matching_end_date'] = np.max(ip_df['date_var'])

        # Filter down to only the top matches
        distances_df = distances_df[distances_df['Skip'] == False]
        distances_df['dist_rank'] = distances_df['RelativeDistance'].rank(method='first', ascending=True)
        distances_df['corr_rank'] = distances_df['Correlation'].rank(method='first', ascending=False)
        distances_df['combined_rank'] = (distances_df['w']*distances_df['dist_rank']
                                        + (1-distances_df['w'])*distances_df['corr_rank'])
        distances_df = distances_df.sort_values('combined_rank', ascending=True)
        distances_df.drop(['dist_rank', 'Skip', 'combined_rank', 'corr_rank'], axis=1, inplace=True)
        distances_df['rank'] = np.arange(distances_df.shape[0])
        distances_df = distances_df[distances_df['rank'] <= matches]
        distances_df.drop(['matches', 'w'], axis=1, inplace=True)

        #TODO tidyr::replace_na(list(Correlation_of_logs=0)) line 123

        if dtw_emphasis == 0 and distances_df.shape[0] > 0:
            distances_df['RelativeDistance'] = None

        return distances_df


    def stop_if(value, clause, message):
        """
        TODO replace with a Python equivalent
        """
        if value == clause:
            print(message)

            #TODO put some version of stopping the execution



    def best_matches(data=None, markets_to_be_matched=None, id_variable=None, date_variable=None,
                    matching_variable = None, parallel=False, warping_limit=1, start_match_period= None,
                    end_match_period=None, matches=None, dtw_emphasis=1, suggest_market_splits=False, split_bins=20):

        """
        I've used `ip_df` instead of `data` in R

        """

        # Check the start date and end dates
        try:
            start_match_period is None
        except:
            print('No start date provided')

        try:
            end_match_period is None
        except:
            print('No end date provided')


        # clean up the emphasis
        if dtw_emphasis is None:
            dtw_emphasis = 1
        elif dtw_emphasis > 1:
            dtw_emphasis = 1
        elif dtw_emphasis < 0:
            dtw_emphasis = 0


        # TODO check the inputs

        # rename columns to match the ones used in the R package
        ip_df = data.copy(deep=True)
        col_map = {date_variable: 'date_var', id_variable: 'id_var', matching_variable: 'match_var'}
        ip_df = ip_df.rename(columns = col_map)

        if matches is None:
            if (markets_to_be_matched is None and suggest_market_splits == True):
                matches = ip_df['id_var'].nunique
            else:
                matches = 5
        else:
            if (markets_to_be_matched is None and suggest_market_splits == True):
                matches = ip_df['id_var'].nunique
                print("The matches parameter has been overwritten to conduct a full search for optimized pairs")


        # check for dupes
        ddup_df = ip_df.drop_duplicates()

        try:
            ddup_df.shape[0] < ip_df.shape[0]
        except:
            print('ERROR: There are date/market duplicates in the input data')


        ##TODO reduce the width of the data.frame

        saved_data = ip_df.copy(deep=True)

        # filter the dates
        #TODO check the rest of this block
        ip_df = ip_df[(ip_df['date_var'] >= start_match_period) &  (ip_df['date_var'] <= end_match_period)]


        ## TODO use the right kind of exception
        # check if any data is left

        try:
            ip_df.shape[0] == 0
        except:
            print("ERROR: no data left after filter for dates")


        # get a vector of all markets that matches are wanted for.
        # Check to ensure markets_to_be_matched exists in the data.

        segmentation = False

        if markets_to_be_matched is None:

            markets_to_be_matched = list(ip_df['id_var'].unique())
            segmentation = True
        else:
            markets_to_be_matched = list(set(markets_to_be_matched))

            #TODO warning in case of invalid test market values

        # loop through markets and compute distances

        if not parallel:
            for i in range(len(markets_to_be_matched)):
                dist_op = R_MarketMatching.calculate_distances(markets_to_be_matched, ip_df, id_variable,
                                              i, warping_limit, matches, dtw_emphasis
                                              )

                if i == 0:
                    # initialized all_distances as dataframe instead of list
                    all_distances = dist_op.copy(deep=True)
                else:
                    all_distances = all_distances.append(dist_op, ignore_index=True)

            shortest_distances_df = all_distances.copy(deep=True)

        else:
            #TODO add parallel execution
            print("Parallel execution has not been added yet - executing sequentially")
            
            for i in range(len(markets_to_be_matched)):
                dist_op = R_MarketMatching.calculate_distances(markets_to_be_matched, ip_df, id_variable,
                                                i, warping_limit, matches, dtw_emphasis
                                                )

                if i == 0:
                    # initialized all_distances as dataframe instead of list
                    all_distances = dist_op.copy(deep=True)
                else:
                    all_distances = all_distances.append(dist_op, ignore_index=True)

            shortest_distances_df = all_distances.copy(deep=True)

        suggested_split = None
        #TODO check the next part if segmentation == True - this part has changed in the most recent version
        if segmentation:

            sizes = shortest_distances_df.copy()
            sizes['market'] = sizes[id_variable] # check this part
            markets = sizes['market'].nunique
            max_bins = np.floor(markets/2)
            bins = max_bins.copy()

            if max_bins > split_bins:
                bins = split_bins
                if bins == 0:
                    bins = 1
                elif max_bins == 0:
                    bins = 1

            sizes = sizes[['market', 'SUMTEST']]
            sizes = sizes.drop_duplcates('market')
            sizes['rank'] = sizes['SUMTEST'].rank(method='first', ascending=False) # check the ascending parameter
            sizes['DECILE'] = pd.qcut(sizes['rank'], bins)
            sizes = sizes[['market', 'DECILE']]

            sizes_grouped = sizes.groupby(["DECILE"])
            [sizes_grouped.get_group(group) for group in sizes_grouped.groups]

            optimal_list = []
            j = 1
            for i in range(len(sizes_grouped)): #TODO check
                bin_ = sizes_grouped.at(i, market)
                tdf = shortest_distances_df.copy()
                tdf['test_market'] = tdf[id_variable]
                tdf = tdf[(tdf['test_market'] == bin_ & tdf['BestControl'] == bin_)]
                tdf = tdf.sort_values('Correlation_of_logs', ascending=False)
                tdf['control_market'] = tdf['BestControl']
                tdf['Segment'] = i

                rows_left = tdf.shape[0]

                while rows_left > 1:
                    # check these column names
                    col_list = ["Segment", "test_market", "control_market", "Correlation_of_logs", "SUMTEST", "SUMCNTL"]
                    optimal_list[j] = tdf[0, col_list]
                    test = tdf.at[0, 'test_market']
                    cntl = tdf.at[0, 'control_market']

                    tdf = tdf[~(tdf['test_market'].isin(test, cntl))] #check the use of isin
                    tdf = tdf[~(tdf['control_market'].isin(test, cntl))] #check the use of isin
                    tdf = tdf.sort_values('Correlation_of_logs', ascending=False)
                    rows_left = tdf.shape[0]
                    j += 1

                suggested_split = pd.DataFrame(optimal_list)
                suggested_split = suggested_split.sort_values('Segment')
                suggested_split = suggested_split.sort_values('Correlation_of_logs', ascending=False)

                #TODO fix this line 422 in R
                # suggested_split['pair_rank'] = suggested_split[''].rank()
                suggested_split['pair_rank'] = np.arange(suggested_split.shape[0])

                suggested_split['v'] = suggested_split['SUMTEST'] + suggested_split['SUMCNTL']
                suggested_split['percent_of_volume'] = suggested_split['v'].cumsum()/suggested_split['v'].sum()
                suggested_split.drop('v', inplace=True)


        # Return the results
        output = {}
        output["best_matches"] = shortest_distances_df
        output["data"] = saved_data
        output["market_id"] = id_variable
        output["matching_metric"] = matching_variable
        output["date_variable"] = date_variable
        output["suggested_test_control_splits"] = suggested_split

        return output


    def inference(matched_markets=None, bsts_modelargs=None, test_market=None, end_post_period=None,
                alpha=0.05, prior_level_sd=0.01, control_matches=5, analyze_betas=False, nseasons=None
                ):


        try:
            len(test_market) > 1
        except:
            print("ERROR: inference() can only analyze one test market at a time. \
                Call the function separately for each test market")

        if bsts_modelargs is not None and nseasons is not None:
            print("NOTE: You're passing arguments directly to bsts while also specifying nseasons")
            print("NOTE: bsts_modelargs will overwrite nseasons")

        if not bsts_modelargs:
            if not nseasons:
                bsts_modelargs = {"prior_level_sd": prior_level_sd}
            else:
                bsts_modelargs = {"nseasons": nseasons, "prior_level_sd": prior_level_sd}
        else:
            if analyze_betas:
                analyze_betas = False
                print("NOTE: analyze_betas turned off when bsts model arguments are passed directly")
                print("Consider using the nseasons and prior_level_sd parameters instead")

        # copy the distances
        best_matches = matched_markets["best_matches"]
        mm = best_matches[best_matches['rank'] <= control_matches]

        data = matched_markets["data"]

        mm['id_var'] = mm.iloc[:,0]
        mm.sort_values(['id_var', 'BestControl'], ascending = [True, True], inplace=True)

        # TODO convert to try catch
        ## check if the test market exists
        if test_market not in data['id_var'].unique():
            print("ERROR: Test market " + test_market + " does not exist")

        ## if an end date has not been provided, then choose the max of the data
        if not end_post_period:
            filt_df = data[data['id_var'] == test_market]
            end_post_period = filt_df['date_var'].max()

        # filter for dates
        filt_df = mm[mm['id_var'] == test_market] # why is this required?
        matching_start_date = filt_df['matching_start_date'].iloc[0]
        matching_end_date = filt_df['matching_end_date'].iloc[0]

        data = data[(data['date_var'] >= matching_start_date) & (data['date_var'] <= end_post_period)]

        ## get the control market name
        filt_df = mm[mm['id_var'] == test_market]

        control_market = list(filt_df['BestControl'])

        ## get the test and ref markets
        mkts = R_MarketMatching.create_market_vectors(data, test_market, control_market)
        y = list(mkts.iloc[:,1])
        ref = list(mkts.iloc[:,2])
        date = list(mkts.iloc[:,0])

        end_post_period = max(date)
        post_period = list(filter(lambda x: (x > mm['matching_end_date'].iloc[0]), date))

        # TODO check the exception
        if len(post_period) == 0:
            raise Exception('ERROR: no valid data in the post period')

        post_period_start_date = min(post_period)
        post_period_end_date = max(post_period)

        ts = pd.DataFrame(list(zip(y, date, ref)), columns =['y', 'date', 'ref'])

        #TODO check if this is what "zoo" does
        ts.set_index('date', inplace=True)
        ts.sort_values(['date'], inplace=True)

        # print the settings
        print('------------- Inputs -------------')
        print(f'Test Market: {test_market}')

        for i in range(len(control_market)):
            print(f'Control Market {i}: {control_market[i]}')

        #TODO print from Line 617

        model_parms = bsts_modelargs.keys()

        for key, value in bsts_modelargs.items():
            print(f"{key} : {value}")

        if nseasons not in model_parms:
            print('No seasonality component (controlled for by the matched markets) ')

        print(f'Posterior Intervals Tail Area: {100*(1 - alpha)}')

        pre_period = [matching_start_date, matching_end_date]
        post_period = [post_period_start_date, post_period_end_date]


        #TODO add alpha=alpha, model.args=bsts_modelargs
        impact = CausalImpact(ts, pre_period, post_period)

        #TODO implement analyze_betas = True
        if analyze_betas:
            print('analyze betas not implemented yet')


        #TODO burn <- SuggestBurn(0.1, impact$model$bsts.model)

        #TODO Line 669 from ci.summary
        ## create statistics

        results = {}

        impact_summ_data = impact.summary_data
        impact_plot = impact.plot

        results['causal_impact_object'] = impact
        results['summary_data'] = impact_summ_data
        results['all_output_plots'] = impact_plot

        ## compute mape

        print('------------- Model Stats -------------')

        print('Matching (pre) Period MAPE: ')

        return results

        
