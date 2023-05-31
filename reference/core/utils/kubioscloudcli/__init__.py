import os

import requests
from dotenv import load_dotenv

curr_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(curr_dir, '.env'))


INVALID_VALUE = -9999


def _get_cognito_access_token(
    client_id,
    client_secret,
    deployment='stg',
    user=None,
    scopes=None,
    aws_region='eu-west-1',
):
    """_summary_
    Parameters
    ----------
    deployment : _type_
        _description_
    user : _type_
        _description_
    aws_region : _type_
        _description_
    client_id : _type_
        _description_
    client_secret : _type_
        _description_
    scopes : _type_
        _description_
    Returns
    -------
    _type_
        _description_
    Raises
    ------
    ValueError
        _description_
    """
    if deployment == "prod":
        pool_name = "kubioscloud"
    elif deployment == "stg":
        pool_name = "kubioscloud-stg"
    elif deployment == "dev" and user:
        pool_name = "kubioscloud-{}".format(user)
    else:
        raise ValueError

    url = "https://{}.auth.{}.amazoncognito.com/oauth2/token".format(
        pool_name, aws_region
    )

    data = {"client_id": client_id, "grant_type": "client_credentials"}
    if scopes:
        data["scope"] = " ".join(scopes.split(","))

    print(
        "Authenticating to '{}' with client_id: {}".format(url, client_id)
    )
    response = requests.post(
        url=url, data=data, auth=(client_id, client_secret)
    )

    if response:
        print("Authentication successful.")
        return response.json()["access_token"], None
    else:
        return None, response.json()["error"]


def _api_call(reqs, verb, path, deployment='stg', user=None, params=None, data=None):
    if deployment == "dev":
        api_base_url = "https://analysis.{}.dev.kubioscloud.com".format(
            user)
    elif deployment == "stg":
        api_base_url = "https://analysis.stg.kubioscloud.com"
    elif deployment == "prod":
        api_base_url = "https://analysis.kubioscloud.com"
    else:
        raise ValueError

    response = reqs.request(verb, url=api_base_url +
                            path, params=params, json=data)

    if response:
        return response.json(), None
    else:
        output_json = response.json()
        return None, output_json.get("error", output_json)


def do_analytics_readiness(data, reqs):
    """_summary_
    Parameters
    ----------
    data : _type_
        _description_
    reqs : _type_
        _description_
    Returns
    -------
    _type_
        _description_
    """
    data = {"type": "RRI", "data": data,
            "analysis": {
                "type": "readiness",
                "history": []}}

    verb = "POST"
    path = "/v2/analytics/analyze"
    return _api_call(reqs, verb, path, data=data)


def predict(RR_list):
    """_summary_
    Parameters
    ----------
    RR_list : _type_
        _description_
    """
    # Call Cognito for access token
    client_id = os.getenv('KUBIOS_CLIENT_ID')
    client_secret = os.getenv('KUBIOS_CLIENT_SECRET')
    user = os.getenv('KUBIOS_USER', 'bonfire')
    apikey = os.getenv('KUBIOS_API_KEY')
    deployment = os.getenv('KUBIOS_ENVIRONMENT', 'stg')
    access_token, error = _get_cognito_access_token(
        deployment=deployment,
        user=user,
        client_id=client_id,
        client_secret=client_secret
    )
    if error:
        raise Exception("Authorization failed: {}".format(error))

    # Construct a requests session which has the required auth headers in place already
    headers = {
        "Authorization": "Bearer {}".format(access_token),
        "X-Api-Key": apikey,
    }
    reqs = requests.Session()
    reqs.headers.update(headers)

    try:
        print(f'RR_list: {RR_list}')
        results = do_analytics_readiness(RR_list, reqs)
        print(f'Kubios Cloud results: {results}')
        if results[0] is not None:
            return results[0]['analysis']['sns_index']
        else:
            print('❌ Failed to invoke Kubios\'s api. Returned None')
            return INVALID_VALUE
    except Exception as e:
        print(f'❌ Failed to analytics. Error: {e}')
        return INVALID_VALUE
