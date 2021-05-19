"""
    Auction Emulator to generate bid requests from iPinYou DataSet.
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import configparser
import math
import json
import os
import pandas as pd

class AuctionEmulatorEnv(gym.Env):
    """
    AuctionEmulatorEnv can be used with Open AI Gym Env and is used to generate
    the bid requests reading the iPinYou dataset files.
    Toy data set with 100 lines are included in the data directory.
    """
    metadata = {'render.modes': ['human']}

    def _load_config(self):
        """
        Parse the config.cfg file
        """
        cfg = configparser.ConfigParser(allow_no_value=True)
        env_dir = os.path.dirname(__file__)
        cfg.read(env_dir + '/config.cfg')
        self.data_src = cfg['data']['dtype']
        if self.data_src == 'ipinyou':
            self.file_in = str(cfg['data']['ipinyou_path'])
        self.metric = str(cfg['data']['metric'])

    def _load_test_config(self):
        cfg = configparser.ConfigParser(allow_no_value=True)
        env_dir = os.path.dirname(__file__)
        cfg.read(env_dir + '/config.cfg')
        self.data_src = cfg['data']['dtype']
        if self.data_src == 'ipinyou':
            self.file_in = str(cfg['data']['ipinyou_path_test'])
        self.metric = str(cfg['data']['metric'])

    def __init__(self):
        """
        Args:
        Populates the bid requests to self.bid_requests list.
        """
        self._load_config()
        self._step = 1
        fields =    [
                    'click',
                    'weekday',
                    'hour',
                    # 'auction_type',
                    'bidprice',
                    'slotprice',
                    'payprice',
                    'click_prob'
                    ]
        dtype = {'click':int, 'weekday': int, 'hour': int, 'bidprice': int, 'slot_price': int, 'payprice': int, 'click_prob': float}
        self.bid_requests = pd.read_csv(self.file_in, sep=',', usecols=fields, dtype=dtype)
        self.total_bids = len(self.bid_requests)
        self.bid_line = {}

    def test_init(self):
        self._load_test_config()
        self._step = 1
        fields = [
            'click',
            'weekday',
            'hour',
            # 'auction_type',
            'bidprice',
            'slotprice',
            'payprice',
            'click_prob'
        ]
        dtype = {'click':int, 'weekday': int, 'hour': int, 'bidprice': int, 'slot_price': int, 'payprice': int, 'click_prob': float}
        self.bid_requests = pd.read_csv(self.file_in, sep=',', usecols=fields, dtype=dtype)
        self.total_bids = len(self.bid_requests)
        self.bid_line = {}

    def _get_observation(self, bid_req):
        observation = {}
        if bid_req is not None:
            observation['click'] = bid_req['click']
            observation['weekday'] = bid_req['weekday']
            observation['hour'] = bid_req['hour']
            # observation['auction_type'] = bid_req['auction_type']
            observation['slotprice'] = bid_req['slotprice']
            observation['click_prob'] = bid_req['click_prob']
        return observation

    def _bid_state(self, bid_req):
        self.click = bid_req['click']
        # self.auction_type = bid_req['auction_type']
        self.bidprice = bid_req['bidprice']
        self.payprice = bid_req['payprice']
        self.click_prob = bid_req['click_prob']
        self.slotprice = bid_req['slotprice']

    def reset(self):
        """
        Reset the OpenAI Gym Auction Emulator environment.
        """
        self._step = 1
        bid_req = self.bid_requests.iloc[self._step]
        self._bid_state(bid_req)
        # observation, reward, cost, done
        return self._get_observation(bid_req), 0.0, 0.0, False

    def step(self, action,budget):
        """
        Args:
            action: bid response (bid_price)
        Reward is computed using the bidprice to payprice difference.
        """
        done = False
        r = 0.0 # immediate reward
        r_p = 0.0 # temp reward
        c = 0.0 # cost for the bid impression

        if self.metric == 'clicks':
            r_p = self.click
        else:
            raise ValueError(f"Invalid metric type: {self.metric}")

        # mkt_price = max(self.slotprice, self.payprice)
        # rctr = 0.008
        # # gate = -4.18404795e-03 + rctr * 9.31356664e+00/ math.log(budget) 499 476 480
        # gate = 0.1060081/math.log(budget) - 0.77394169*rctr
        # if self.click_prob > gate:
        mkt_price = self.payprice
        if action > mkt_price:
            # if self.auction_type == 'SECOND_PRICE':
            r = r_p
            c = mkt_price
                # elif self.auction_type == 'FIRST_PRICE':
                # r = r_p
                # c = action
                # else:
                #     raise ValueError(f"Invalid auction type: {self.auction_type}")

        next_bid = None
        if self._step < self.total_bids - 1:
            next_bid = self.bid_requests.iloc[self._step]
            self._bid_state(next_bid)
        else:
            done = True

        self._step += 1

        return self._get_observation(next_bid), r, c, done

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass
