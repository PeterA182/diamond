import os
import gc
import sys
import numpy as np
import pandas as pd
import utilities as util
CONFIG = util.load_config()


class diamond(object):
    """
    Class for handling relationships between normalized tables pulled from API

    Standardizing adding starting pitchers, lineups (expected and/or actual)

    Adding pitcher rolling stats
    Adding batter rolling stats

    """
    
    def __init__(self, seasonKey=None, min_date_gte=None, max_date_lte=None):
        self.seasonKey = seasonKey
        self.league = 'mlb'
        self.min_date_gte = min_date_gte
        self.max_date_lte = max_date_lte

        # Pitching Stats attributes
        self.pitching_roll_windows = [1, 3, 5, 10]
        self.pitching_stats = ['fip', 'bb_per9', 'hr_fb_ratio', 'k_per9', 'gbpct']
        self.pitching_roll_stats = [
            '{}_roll{}'.format(s, w) for s in self.pitching_stats for
            w in self.pitching_roll_windows
        ]

        # Batting Stats Attributes
        self.batting_roll_windows = [1, 3, 5, 10]
        self.batting_stats = []
        
        # Check args
        assert not (
            seasonKey and
            (min_date_gte != None) and
            (max_date_lte != None)
        )
        
        # Determine time period
        if self.seasonKey:
            self.min_date_gte = CONFIG.get(self.league)\
                .get('seasons')\
                .get(self.seasonKey)\
                .get('seasonStart')
            self.max_date_lte = CONFIG.get(self.league)\
                .get('seasons')\
                .get(self.seasonKey)\
                .get('seasonEnd')

        # Read in from daily game
        path = CONFIG.get(self.league)\
            .get('paths')\
            .get('normalized').format(
                f='daily_games'
            )
        paths = [
            path+fname for fname in os.listdir(path) if (
                (fname[:8] >= self.min_date_gte)
                &
                (fname[:8] <= self.max_date_lte)
            )
        ]
        self.summary = pd.concat(
            objs=[pd.read_parquet(p) for p in paths],
            axis=0
        )
        self.summary.drop_duplicates(subset=['gameId'], inplace=True)

        self.summary.loc[:, 'gameStartDate'] = \
            pd.to_datetime(self.summary['startTime'].str[:10])
        

    def add_starting_pitchers(self, dispositions=['home', 'away']):
        """
        """

        # Paths
        atbats_path = CONFIG.get(self.league)\
            .get('paths')\
            .get('normalized').format(
                f='game_atbats'
            )
        atbats_paths = [atbats_path+d+"/" for d in os.listdir(atbats_path) if (
            (d >= self.min_date_gte)
            &
            (d <= self.max_date_lte)
        )]
        atbats_paths_full = []
        for abp in atbats_paths:
            atbats_paths_full.extend([abp+fname for fname in os.listdir(abp)])

        # Get atbats 
        df_ab = pd.concat(
            objs=[pd.read_parquet(p) for p in atbats_paths_full],
            axis=0
        )

        # Get Home Starters
        df_top1 = df_ab.loc[(
            (df_ab['inning']==1) &
            (df_ab['inningHalf']=='TOP') &
            (df_ab['outCount']==0)
        ), :]
        df_home_starters = df_top1.loc[:, ['gameId', 'pitcherId']]\
            .drop_duplicates(subset=['gameId'])
        df_home_starters.rename(
            columns={'pitcherId': 'homeStartingPitcherId'},
            inplace=True
        )

        # Get Away Starters
        df_bot1 = df_ab.loc[(
            (df_ab['inning']==1) &
            (df_ab['inningHalf']=='BOTTOM') &
            (df_ab['outCount']==0)
        ), :]
        df_away_starters = df_bot1.loc[:, ['gameId', 'pitcherId']]\
            .drop_duplicates(subset=['gameId'])
        df_away_starters.rename(
            columns={'pitcherId': 'awayStartingPitcherId'},
            inplace=True
        )

        # Assemble starters
        df_starters = pd.merge(
            df_home_starters, 
            df_away_starters, 
            how='outer', 
            on=['gameId'], 
            validate='1:1'
        )

        # Merge
        self.summary = pd.merge(
            self.summary,
            df_starters,
            how='left',
            on=['gameId'],
            validate='1:1'
        )

        
    def add_pitcher_rolling_stats(
            self, pitcher_cols=['homeStartingPitcherId', 'awayStartingPitcherId'],
            shift_back=True
    ):
        """
        """

        # Path
        ptch_roll_path = CONFIG.get(self.league)\
            .get('paths')\
            .get('rolling_stats').format('pitching')+"player/"

        # Read in
        ptch_roll = pd.concat(
            objs=[pd.read_parquet(ptch_roll_path+fname) for fname in
                  os.listdir(ptch_roll_path) if 
                 ((fname.replace(".parquet", "") >= self.min_date_gte)
                  &
                  (fname.replace(".parquet", "") <= self.max_date_lte))],
            axis=0
        )

        # Create rolling metrics
        cols = ['gameId', 'gameStartDate', 'playerId'] +\
            self.pitching_roll_stats
        for mt in cols:
            if mt not in ptch_roll.columns:
                print("MISSING: {}".format(mt))

        # Subset
        ptch_roll = ptch_roll.loc[:,
            ['gameId', 'gameStartDate', 'playerId'] +
            self.pitching_roll_stats
        ]

        # Sort
        ptch_roll.sort_values(by=['gameStartDate'], ascending=True, inplace=True)

        # Shift back if interested in rolling stats leading up to game
        if shift_back:
            for col in self.pitching_roll_stats:
                msk = (ptch_roll['playerId'].shift(1)==ptch_roll['playerId'])
                ptch_roll.loc[msk, col] = ptch_roll[col].shift(1)

        # Handle Infs
        for col in self.pitching_roll_stats:
            ptch_roll = ptch_roll.loc[~ptch_roll[col].isin([np.inf, -np.inf]), :]

        # Prep self.starting_pitcher_stats
        self.starting_pitcher_stats = \
            self.summary[['gameId'] + pitcher_cols]
            
        # Merge back to starters (one at a time)
        # Home
        if 'homeStartingPitcherId' in pitcher_cols:
            df = ptch_roll.rename(
                columns={c: 'homeStarter_{}'.format(c) for
                         c in self.pitching_stats},
                inplace=False)
            self.starting_pitcher_stats = pd.merge(
                self.starting_pitcher_stats,
                df,
                how='left',
                left_on=['gameId', 'homeStartingPitcherId'],
                right_on=['gameId', 'playerId'],
                validate='1:1'
            )
            self.starting_pitcher_stats.drop(labels=['playerId'],
                                             axis=1,
                                             inplace=True)

        # Away
        if 'awayStartingPitcherId' in pitcher_cols:
            df = ptch_roll.rename(
                columns={c: 'awayStarter_{}'.format(c) for
                         c in self.pitching_stats},
                inplace=False)
            self.starting_pitcher_stats = pd.merge(
                self.starting_pitcher_stats,
                df,
                how='left',
                left_on=['gameId', 'awayStartingPitcherId'],
                right_on=['gameId', 'playerId'],
                validate='1:1'
            )
            self.starting_pitcher_stats.drop(labels=['playerId'],
                                             axis=1,
                                             inplace=True)
            

    def add_lineups(self, status='auto'):
        """
        status: 'auto' - expected/actual
        """

        None


    def add_batter_rolling_stats(self, shift_back=True):
        """
        """

        None


    def fit_batter_cluster_model(self, k):
        """
        """

        None


    def fit_pitcher_cluster_model(self, k):
        """
        """

        None
        
        
if __name__ == "__main__":

    
    print("Instantiate")
    d = diamond(seasonKey='s2019')
    print("Adding starting pitchers")
    d.add_starting_pitchers()
    print("Adding pitcher rolling stats")
    d.add_pitcher_rolling_stats()
