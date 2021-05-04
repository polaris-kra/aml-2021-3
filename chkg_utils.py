import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse
from scipy.stats import spearmanr, kendalltau, beta


def load_players(fpath):
    players = pd.read_pickle(fpath)
    players = pd.DataFrame(list(players.values()))
    players['name'] = players['name'] + ' ' + players['surname']
    players = players[['id', 'name']]
    return players


def load_tournaments(fpath):
    tournaments = pd.read_pickle(fpath)
    tournaments = pd.DataFrame(list(tournaments.values()))
    tournaments['year'] = pd.to_datetime(tournaments['dateStart']).apply(pd.to_datetime, utc=True).dt.year
    tournaments['type'] = tournaments['type'].apply(lambda t: t['name'])
    tournaments = tournaments[['id', 'name', 'year', 'type']].reset_index(drop=True)
    return tournaments


def load_results(fpath, tournaments):
    results = pd.read_pickle(fpath)
    team_results = []

    for tournament_id, teams in tqdm(results.items()):
        if tournament_id not in tournaments['id'].values:
            continue
        if (len(teams) < 2):
            continue

        tournament_results = []
        for team in teams:
            if ('mask' not in team) or \
               (team['mask'] is None) or \
               ('X' in team['mask']) or \
               ('?' in team['mask']) or \
               (len(team['teamMembers']) == 0):
                continue

            maks = np.array(list(map(int, team['mask'].replace('X', '0').replace('?', '0'))))

            team_result = {}
            team_result['game_id'] = tournament_id
            team_result['team_id'] = team['team']['id']
            team_result['questions'] = maks
            team_result['wins'] = sum(maks)
            team_result['total'] = len(maks)
            team_result['position'] = team['position']
            team_result['members'] = [m['player']['id'] for m in team['teamMembers']]
            team_result['num_players'] = len(team_result['members'])

            tournament_results.append(team_result)
            
        if len(set([r['total'] for r in tournament_results])) == 1:
            team_results.extend(tournament_results)

    team_results = pd.DataFrame(team_results)
    team_results['year'] = team_results.merge(tournaments, left_on='game_id', right_on='id')['year']

    return team_results


def correlation_score(y_game):
    y_true = y_game['position'].values
    y_pred = y_game['team_rating'].values
    return pd.Series({'spearman': -spearmanr(y_true, y_pred)[0], 
                      'kendall': -kendalltau(y_true, y_pred)[0]})
    
    
def mean_correlation_score(y_true, y_pred):
    y_games = y_true.merge(y_pred, on=['game_id', 'team_id'])
    y_stats = y_games.groupby('game_id').apply(correlation_score).dropna()
    score = y_stats.values.mean(axis=0)
    return {'spearman': round(score[0], 3), 'kendall': round(score[1], 3)}


def get_question_ids(g):
    game_id = g['game_id'].values[0]
    questions_mask = np.array(list(g['questions'].values))
    question_ids = list(map(lambda x: list( 1000*game_id + np.where(x >  0)[0] + 1) + 
                                      list(-1000*game_id - np.where(x == 0)[0] - 1), questions_mask))
    return question_ids


def prepare_data(data, filter_players=None):
    data = data[['game_id', 'team_id', 'members', 'questions']].copy()
    data['win_questions'] = data.groupby('game_id').apply(get_question_ids).explode().values
    data['gameteam_id'] = 10000*data['team_id'] + data['game_id']
    data = data.drop(['game_id', 'team_id', 'questions'], axis=1).reset_index(drop=True)
    data.columns = ['player_id', 'question_id', 'gameteam_id']
    
    data = data.explode('player_id').explode('question_id').reset_index(drop=True)
    if filter_players is not None:
        data = data[data['player_id'].isin(filter_players)]
    
    target = (data['question_id'] > 0).astype(int).values
    data['question_id'] = abs(data['question_id'])
    data['win'] = target

    players = data['player_id'].unique()
    questions = data['question_id'].unique()
    player_map = {pid: idx for idx, pid in enumerate(players)}
    question_map = {qid: idx for idx, qid in enumerate(questions)}

    player_inv_map = {idx:pid for pid,idx in player_map.items()}
    question_inv_map = {idx:qid for qid,idx in question_map.items()}
    
    data['player_id'] = data['player_id'].map(player_map)
    data['question_id'] = data['question_id'].map(question_map)

    return data, player_inv_map, question_inv_map













def top_score(top100_true, top100_pred):
    players_true = set(top100_true['player_id'].values)
    players_pred = set(top100_pred['player_id'].values)
    players_common = players_pred.intersection(players_true)
    players_all = players_pred.union(players_true)
    
    players_join = top100_true.merge(top100_pred, on='player_id')
    y_true = players_join['player_rating_x'].values
    y_pred = players_join['player_rating_y'].values
    iou = len(players_common)/len(players_all)
    
    iou_score = iou
    corr_score = (spearmanr(y_true, y_pred)[0] + kendalltau(y_true, y_pred)[0])*0.25 + 0.5
    
    score = 0.8*iou_score + 0.2*corr_score

    return round(score, 5)


def get_player_questioncount(data):
    data = data[['members', 'questions']].copy()
    data['win_count'] = data['questions'].apply(len)
    data = data[['members', 'win_count']]
    data = data.explode('members')
    data.columns = ['player_id', 'question_count']
    data['player_id'] = data['player_id'].astype(int)
    data = data.groupby('player_id').agg({'question_count': 'sum'}).reset_index()
    
    return data
