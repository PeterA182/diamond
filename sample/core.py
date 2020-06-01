import os
import gc
import sys
import warnings
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
        ADDS DIMENSIONS TO SUMMARY 
        ADDS ATTRIBUTE ".starters"
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


    def add_bullpen_summary(self, dispositions=['home', 'away']):
        """
        ADDS ATTRIBUTE "bullpens_summary"
        """

        # Get atbats, filter to where not equal to starters
        if not all(
            s in self.summary.columns for s in \
           ['{}StartingPitcherId'.format(d) for d in dispositions]
        ):
            self.add_starting_pitchers()
        
        # Get atbats
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

        # Get atbats and sort by inning / outCount
        df_ab = pd.concat(
            objs=[pd.read_parquet(p) for p in atbats_paths_full],
            axis=0
        )
        df_ab = df_ab.loc[:, ['gameId', 'pitcherId', 'inning', 'inningHalf', 'outCount']]
        
        # Select home, sort, dd, remove starter, and rerank
        bullpen_summary = []
        sides = {'TOP': 'home', 'BOTTOM': 'away'}
        for half_, disp  in sides.items():

            # Set up starter map for later mask
            startingPitcherMap = self.summary.set_index('gameId')\
                ['{}StartingPitcherId'.format(disp)].to_dict()
            
            df_ab_h = df_ab.loc[df_ab['inningHalf']==half_, :]
            # Sort
            df_ab_h = df_ab_h.sort_values(by=['gameId', 'inning', 'outCount'], ascending=True, inplace=False)
            # Drop labels
            df_ab_h = df_ab_h.drop(labels=['inning', 'outCount'], axis=1, inplace=False)
            # Remove pitcher who was already identified as starter (self.summary['homeStartingPitcherId'].iloc[0]?
            df_ab_h.loc[:, '{}StartingPitcherId'.format(disp)] = \
                df_ab_h['gameId'].map(startingPitcherMap)
            df_ab_h = df_ab_h.loc[df_ab_h['pitcherId'] != df_ab_h['{}StartingPitcherId'.format(disp)], :]
            # Handle ordering
            df_ab_h['pitcherAppearOrder'] = df_ab_h.groupby(by=['gameId'])['pitcherId'].rank(method='first')
            df_ab_h = df_ab_h.groupby(by=['gameId', 'pitcherId'], as_index=False).agg({'pitcherAppearOrder': 'min'})
            df_ab_h['pitcherAppearOrder'] = df_ab_h.groupby(by=['gameId'])['pitcherId'].rank(method='first')
            df_ab_h['pitcherAppearOrderMax'] = df_ab_h.groupby('gameId')['pitcherAppearOrder'].transform('max')
            # Label middle pitchers relief role and last pitcher closer` role
            msk = (df_ab_h['pitcherAppearOrder']==df_ab_h['pitcherAppearOrderMax'])
            df_ab_h.loc[msk, 'pitcherRoleType'] = 'closer'
            df_ab_h.loc[~msk, 'pitcherRoleType'] = 'reliever'

            # Subset (TODO add first inning appeared)
            df_ab_h = df_ab_h.loc[:, ['gameId', 'pitcherId', 'pitcherRoleType', 'pitcherAppearOrder']]
            df_ab_h.loc[:, 'pitcherDisp'] = disp
            bullpen_summary.append(df_ab_h)
        bullpen_summary = pd.concat(objs=bullpen_summary, axis=0)
        self.bullpen_summary = bullpen_summary
        

        
    def add_pitcher_rolling_stats(
        self,
        dispositions=['home', 'away'],
        pitcher_roll_types=['starter', 'reliever', 'closer'],
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

        # Check if starter / all designation
        if 'starter' in pitcher_roll_types:
            print("    Adding stats for starters")

            # Check that summary attribute has starting pitchers
            if not any('StartingPitcherId' in col for col in
                       self.summary.columns):
                self.add_starting_pitchers(dispositions=dispositions)

            # Merge back to starters (one at a time)
            pitcher_cols = ['{}StartingPitcherId'.format(d) for
                            d in dispositions]

            # Prep self.starting_pitcher_stats
            self.starting_pitcher_stats = \
                self.summary[['gameId'] + pitcher_cols]
            
            for disp in dispositions:           
                df = ptch_roll.rename(
                    columns={stat: '{}Starter_{}'.format(disp, stat) for
                             stat in self.pitching_stats},
                    inplace=False)
                self.starting_pitcher_stats = pd.merge(
                    self.starting_pitcher_stats,
                    df,
                    how='left',
                    left_on=['gameId', '{}StartingPitcherId'.format(disp)],
                    right_on=['gameId', 'playerId'],
                    validate='1:1'
                )
                self.starting_pitcher_stats.drop(labels=['playerId'],
                                                 axis=1,
                                                 inplace=True)

        # Check if reliever / all designation
        bullpen_reconstruct = []
        if 'reliever' in pitcher_roll_types:
            print("    Adding stats for relievers")
            
            # Check attribute (try / except cheaper but less readable)
            if not hasattr(self, 'bullpen_summary'):
                self.add_bullpen_summary(self, dispositions=dispositions)

            # Merge back to relievers in bullpen summary
            msk = (self.bullpen_summary['pitcherRoleType'] == 'reliever')
            bullpen = self.bullpen_summary.loc[msk, :]
            #hold = self.bullpen_summary.loc[~msk, :]
            #bullpen_recon = []
            if bullpen.shape[0] == 0:
                warnings.warn("    No relief pitchers found in bullpen_summary attribute")
            
            if not all(d in dispositions for d in ['home', 'away']):
                assert len(dispositions) == 1 and dispositions[0] in ['home', 'away']
            for disp in dispositions:
                bullpen_disp = bullpen.loc[bullpen['pitcherDisp'] == disp, :]
                #df = ptch_roll.rename(
                #    columns={stat: '{}Reliever_{}'.format(disp, stat) for
                #             stat in self.pitching_stats},
                #    inplace=False
                #)
                bullpen_disp = pd.merge(
                    bullpen_disp,
                    ptch_roll,
                    #df,
                    how='left',
                    left_on=['gameId', 'pitcherId'],
                    right_on=['gameId', 'playerId'],
                    validate='1:1'
                )
                bullpen_disp.drop(labels=['playerId'], axis=1, inplace=True)
                bullpen_reconstruct.append(bullpen_disp)
#            bullpen_recon = pd.concat(objs=bullpen_recon, axis=0)
            
#            self.bullpen_summary = bullpen_recon

        # TODO FIX CLOSER MERGE _x _y 
        if 'closer' in pitcher_roll_types:
            print("    Adding stats for closers")

            # Check if closer / all designation
            if not hasattr(self, 'bullpen_summary'):
                self.add_bullpen_summary(self, dispositions=dispositions)

            # Merge back to closers in bullpen summary
            msk = (self.bullpen_summary['pitcherRoleType'] == 'closer')
            bullpen = self.bullpen_summary.loc[msk, :]
            #hold = self.bullpen_summary.loc[~msk, :]
            #bullpen_recon = [hold]
            if bullpen.shape[0] == 0:
                warnings.warn("    No closing pitchers found in bullpen_summary attribute")

            if not all(d in dispositions for d in ['home', 'away']):
                assert len(dispositions) == 1 and dispositions[0] in ['home', 'away']
            for disp in dispositions:
                bullpen_disp = bullpen.loc[bullpen['pitcherDisp'] == disp, :]
                #df = ptch_roll.rename(
                #    columns={stat: '{}Closer_{}'.format(disp, stat) for
                #             stat in self.pitching_stats},
                #    inplace=False
                #)
                bullpen_disp = pd.merge(
                    bullpen_disp,
                    ptch_roll,
                    #df,
                    how='left',
                    left_on=['gameId', 'pitcherId'],
                    right_on=['gameId', 'playerId'],
                    validate='1:1'
                )
                bullpen_disp.drop(labels=['playerId'], axis=1, inplace=True)
                bullpen_reconstruct.append(bullpen_disp)
        bullpen_reconstruct = pd.concat(objs=bullpen_reconstruct, axis=0)

        self.bullpen_summary = bullpen_reconstruct
                
                
                
            
            

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
    d.add_bullpen_summary()
    print("Adding pitcher rolling stats")
    d.add_pitcher_rolling_stats()
    d.bullpen_summary.to_csv('/Users/peteraltamura/Desktop/bullpen_summary_test.csv', index=False)