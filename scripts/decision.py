# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:48:55 2018

@author: jonathan
"""

class decision(object):
    def __init__(self, hh_dict):
        # Initialize the object
        self.households = hh_dict
    def decide(self):
        # Make a decision and return True or False
        pass
    def utility(self, index):
        u = dict( [ (hh.id, hh.eu_df.eu.loc[index]) for hh in self.households.values() ] )
        return(u)
    def unhappy(self, index):
        utilities = self.utility(index)
        baseline = dict( [ (hh.id, hh.eu_df.eu.iloc[0]) for hh in self.households.values() ] )
        return dict([(hh_id, (utilities[hh_id]))] for hh_id in utilities.keys()
                        if utilities[hh_id] < baseline[hh_id] )
        
    def count_unhappy(self, index):
        return len(self.unhappy(index))

    
class election(decision):
    def __init__(self, hh_dict):
        super.__init__(hh_dict)
    def decide(self):
        
    def decide(self):
        ballots = np.array( [ np.array(hh.vote(), dtype = np.integer) \
                  for hh in self.households.values() ] )
        winner = election.instant_runoff(ballots)
        return (winner, self.utility(winner))

    @staticmethod
    def instant_runoff(ballots):
        majority = ballots.shape[0] * 0.5
        n_choices = ballots.shape[1]
        ballots = pd.DataFrame(ballots)
        vc = ballots[0].value_counts()
        while vc.iloc[0] <= majority:
            vc = ballots[n_choices - 1].value_counts()
            eliminate = vc.index[0]
            blist = [ list(x) for x in list(np.array(ballots)) ]
            for b in blist: b.remove(eliminate)
            ballots = pd.DataFrame(np.array(blist, dtype = np.integer))
            n_choices = ballots.shape[1]
            vc = ballots[0].value_counts()
        return vc.index[0]

class auction(decision):
    def __init__(self, hh_dict):
        super.__init__(hh_dict)
        self.transactions = []
        self.ballots = None
        self.initialize_votes()
    def utility(self, index):
        u = dict( [ (hh.id, hh.eu_df.eu.loc[index]) for hh in self.households.values() ] )
        trans = [t for t in self.transactions if t.year == index]
        for t in trans:
            # print "Transaction: buyer = ", t.buyer_id(), ", seller = ", t.seller_id(), ", year = ", t.year[0], ", price = ", t.price
            u[t.buyer_id()] -= t.price
            u[t.seller_id()] += t.price
        return(u)

    def unhappy(self, index):
        potentially_unhappy = dict([(hh.id, hh.eu_df) for hh in self.households.values() if hh.eu_df.index[0] != index])
        u = self.utility(index)
        utilities = dict([(k, u[k]) for k in potentially_unhappy.keys()])
        baseline = dict( [ (k, potentially_unhappy[k].iloc[0]) for k in potentially_unhappy.keys()])
        unhappy = dict([(k, (utilities[k], potentially_unhappy[k])) \
                        for k in potentially_unhappy.keys() \
                        if utilities[k] < potentially_unhappy[k].iloc[0].eu])
        return unhappy
    
    def vote(self, force = False):
        if self.ballots is None:
            return None
        majority = 0.5 * self.ballots.shape[0]
        votes = pd.DataFrame(self.ballots)[0].value_counts()
        print(votes)
        if votes.iloc[0] > majority:
            return votes.index[0]
        elif force:
            return votes.index[0]
        return None

    def initialize_votes(self):
        self.ballots = np.array( [ hh.vote() for hh in self.households.values() ] )
        self.transactions = []

    def decide(self, max_rounds = 1000):
        global tt, bbids
        self.initialize_votes()
        for round in range(max_rounds):
            # print "Round ", round
            winner = self.vote()
            if winner is not None:
                return (winner, self.utility(winner))
            target = self.vote(force = True)
            # print "Target = ", target
            buyers  = [trans.buyer for trans in self.transactions]
            sellers = [trans.seller for trans in self.transactions]
            buyer_ids = [ b.id for b in buyers ]
            seller_ids = [ s.id for s in sellers ]
            neutral_ids = list(set(self.households.keys()) - set(buyer_ids) - set(seller_ids))
            neutral = [ self.households[hh_id] for hh_id in neutral_ids ]
            for hh in neutral:
                hh.construct_bids(target)
            for hh in buyers:
                purchases = [ t for t in self.transactions if t.buyer_id() == hh.id ]
                hh.construct_bids(target, purchases)
            bids = pd.concat([hh.bids for hh in neutral + buyers])
            bbids = bids.copy()
            transactions = self.bidding_round(bids)
            tt = transactions
            for trans in transactions:
                trans.buyer.wealth -= trans.price
                trans.seller.wealth += trans.price
                self.ballots[trans.seller_id()] = trans.year
            if len(transactions) > 0:
                # print len(transactions), " Transactions"
                self.transactions += transactions
            else:
                # print "No transactions"
                break
        winner = self.vote(force = True)
        return (winner, self.utility(winner))

    def bidding_round(self,bids):
        global pp, wtpp, wtaa, p0, bcc, scc
        transactions = []
        wta = bids[np.logical_not(bids.is_offer.values)][['id', 'year','amount']]
        wtp = bids[bids.is_offer.values][['id', 'year','amount']]
        wtpp = wtp.copy()
        wtaa = wta.copy()
        wta = wta[np.logical_not(wta.duplicated())]
        wtp = wtp[np.logical_not(wtp.duplicated())]
        # wta = wta.drop_duplicates(inplace = True)
        # wtp = wtp.drop_duplicates(inplace = True)
        if wta is None or wta.shape[0] < 2:
            # print "Empty wta"
            return transactions
        if wtp is None or wtp.shape[0] < 2:
            # print "Empty wtp"
            return transactions
        wta = wta.pivot(index = 'id', columns = 'year', values = 'amount')
        wtp = wtp.pivot(index = 'id', columns = 'year', values = 'amount')
        wtpp = wtp.copy()
        wtaa = wta.copy()
        wta_min = wta.min()
        wtp_max = wtp.max()
        for buyer_index in np.random.choice(wtp.index, wtp.shape[0], replace = False):
            buyer = wtp.loc[buyer_index]
            buyer_year = buyer[np.logical_not(buyer.isnull())]
            offer = buyer_year.values[0]
            # print "Buyer_year.index - ", buyer_year.index
            # print "WTA.columns = ", wta.columns
            if buyer_year.index.values in wta.columns.values:
                seller_candidates = wta[buyer_year.index]
            else:
                continue
            seller_candidates = seller_candidates[np.logical_not(pd.isnull(seller_candidates.values))[:,0]]
            seller_candidates = seller_candidates[(seller_candidates.values <= offer)[:,0]]
            if seller_candidates.shape[0] > 0:
                # print "Year ", buyer_year.index[0], ": ", buyer.index[0], " offers ", offer
                # print "     seller_candidates has shape ", seller_candidates.shape
                seller_index = np.random.choice(seller_candidates.index, 1)[0]
                seller = seller_candidates.loc[seller_index]
                accept = seller.values[0]
                # print "Year ", buyer_year.index[0], ": ", buyer.index[0], " offers ", offer, " and ", \
                #        seller.index[0], " will accept ", accept
                if  offer >= accept:
                    price = (offer + accept) / 2.0
                    # print "Offer accepted: price = ", price
                    if True: # self.households[p.buyer].wealth >= price:
                        # print buyer_index, seller_index
                        bh = self.households[buyer_index]
                        sh = self.households[seller_index]
                        tx = transaction(bh, sh, buyer_year.index, price)
                        transactions.append(tx)
                        wtp.drop(buyer_index, inplace = True)
                        wta.drop(seller_index, inplace = True)
                else:
                    # print "Offer rejected."
                    pass
            if wta.shape[0] == 0 or wtp.shape[0] == 0:
                break
        # print len(transactions), " transactions."
        return transactions
