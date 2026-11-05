from src.market_rules.china_a_share import can_hypothetically_buy, can_hypothetically_sell, price_limit_ratio


def test_a_share_constraints_are_centralized():
    assert price_limit_ratio(board="MAIN", st_flag=False, days_since_listing=100) == 0.10
    assert price_limit_ratio(board="STAR", st_flag=False, days_since_listing=100) == 0.20
    assert price_limit_ratio(board="MAIN", st_flag=True, days_since_listing=100) == 0.05
    assert not can_hypothetically_buy(quantity=99, suspended=False, one_word_limit_up=False, at_limit_up=False).allowed
    assert not can_hypothetically_sell(quantity=100, shares_bought_today=100, suspended=False, at_limit_down=False).allowed
