import warnings
warnings.filterwarnings('ignore')

import random
import pandas as pd
from math import gcd

# Create dictionary mapping for integer values of characters
keys = {}
for i, char in enumerate('abcdefghijklmnopqrstuvwxyz0123456789'):
    keys[char] = i + 1

special_characters = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
for i, char in enumerate(special_characters):
    keys[char] = i + 37

# Additive modulus encryption
def am_encrypt(a, k, p):
    a_enc = (a + k) % p
    return a_enc

# Additive modulus decryption
def am_decrypt(a_enc, k, p):
    a_dec = (a_enc - k) % p
    return a_dec

# Generate key pairs for RSA encryption
def generate_key_pair(p, q):
    # Calculate phi(n)
    phi_n = (p - 1)*(q - 1)

    # Calculate public key
    e = random.randint(2, phi_n - 1)
    while gcd(e, phi_n) != 1:
        e = random.randint(2, phi_n - 1)

    # Calculate private key (modulo inverse of public key)
    d = pow(e, -1, phi_n)

    # Return key-pair
    return [e, d]

# RSA encryption
def rsa_encrypt(m, e, n):
    encrypted = []

    # For each character in message perform encryption
    for letter in m.lower():
        # Perform exponential modulus
        enc = keys[letter]**e % n

        # Store encrypted value in string format for dataset
        if enc < 10:
            encrypted.append('0' + str(enc))
        else:
            encrypted.append(str(enc))

    message = ''.join(encrypted)
    return message

# RSA decryption
def rsa_decrypt(encrypted, d, n):
    # Disjoin 2 consecutive index pairs from encrypted string to get each characters encrypted value
    enc_arr = [encrypted[i:i+2] for i in range(0, len(encrypted) - 1, 2)]
    decrypted = []

    # For each encrypted character perform decryption
    for val in enc_arr:
        # Perform exponential modulus
        dec = int(val) ** d % n

        # Store decrypted value in string format for display
        for key, value in keys.items():
            if(dec == value):
                decrypted.append(key)

    winner = ''.join(decrypted)
    return winner

# Main function
def main():
    # list of columns in dataset
    columns_lst = ['auctionid', 'bid', 'bidtime', 'bidder', 'bidderrate', 'openbid', 'price', 'item', 'auction_type']

    # read dataset
    auctions_dataset = pd.read_csv("auction.csv", names = columns_lst, na_values=["?"])

    # Extremely large prime number than max value of bids (Modulus) for Additive Modulus Encryption
    p = 2147483647

    # Random number relatively small to p but extremely large to max bid values (Key) for Additive Modulus Encryption
    k = 94850627

    # Set data types for dataset attributes
    auctions_dataset['auctionid'] = auctions_dataset['auctionid'].astype(str)
    auctions_dataset['bid'] = pd.to_numeric(auctions_dataset['bid'], errors='coerce')
    auctions_dataset['bidtime'] = pd.to_numeric(auctions_dataset['bidtime'], errors='coerce')
    auctions_dataset['bidder'] = auctions_dataset['bidder'].astype(str)
    auctions_dataset['bidderrate'] = pd.to_numeric(auctions_dataset['bidderrate'], errors='coerce')
    auctions_dataset['openbid'] = pd.to_numeric(auctions_dataset['openbid'], errors='coerce')
    auctions_dataset['price'] = pd.to_numeric(auctions_dataset['price'], errors='coerce')
    auctions_dataset['item'] = auctions_dataset['item'].astype(str)
    auctions_dataset['auction_type'] = auctions_dataset['auction_type'].astype(str)

    # Generate key pairs for all users for RSA encryption
    users_col = auctions_dataset['bidder']
    unique_users = users_col[~users_col.str.contains('bidder')].unique()

    # Prime numbers p1 and q values are small due to encryption using large values being computationally expensive and time consuming
    p1 = 7
    q = 11
    n = p1 * q

    kp_dec = {}
    # kp_enc = {}

    # The key-pair generation code is commented due to being computationally expensive and the function is written as a showcase for real-time execution of application
    # for user in unique_users:
    #     kp_dec[user] = generate_key_pair(p1, q)

    kp_dec['schadenfreud'] = [37, 13]
    kp_dec['eli.flint@flightsafety.co'] = [17, 53]

    # Create empty dataframe with required columns (for encrypted values)
    encrypted_column_list = ['auctionid', 'enc_bid', 'enc_bidder', 'item']
    encrypted_dataset = pd.DataFrame(columns = encrypted_column_list)

    # Create empty dataframe for final result dataset
    result_column_list = ['auctionid', 'enc_win_bid', 'enc_win_bidder', 'item']
    result_dataset = pd.DataFrame(columns = result_column_list)
    
    # Encrypt the username of each bidder and his bid and append both values along with product name and auction ID in empty dataframe for encrypted values
    for index, row in auctions_dataset.iterrows():
        # Consider the bid as valid only if it is placed within the time frame of auction deadline and the bid is greater than or equal to the opening bid
        if row['bidtime'] > 0 and row['bidtime'] < float(row['auction_type'][0]) and row['bid'] >= row['openbid']:
            # Perform additive modulus encryption on bids
            encrypted_bid = am_encrypt(row['bid'], k, p)

            # Perform RSA encryption on bidder names
            encrypted_bidder = rsa_encrypt(row['bidder'], kp_dec['eli.flint@flightsafety.co'][0], n)
            # kp_enc[encrypted_bidder] = [kp_dec[row['bidder']][0], kp_dec[row['bidder']][1]]

            encrypted_dataset.at[index, 'auctionid'] = row['auctionid']
            encrypted_dataset.at[index, 'enc_bid'] = encrypted_bid
            encrypted_dataset.at[index, 'enc_bidder'] = encrypted_bidder
            encrypted_dataset.at[index, 'item'] = row['item']

    # Get all the unique auction ids 
    auc_col = auctions_dataset['auctionid']
    auctions_id = auc_col[~auc_col.str.contains('auctionid')].unique()

    # Define iterators
    idx = 0
    i = 0

    # For each unique auction id calculate max bid i.e., the winner
    for id in auctions_id:
        max_bid = 0
        bidder = ''
        item = ''

        # Calculate max bid i.e., the winner
        for index, row in encrypted_dataset.iterrows():
            # Continue to search for current auction id from the row where we left off in the last interation
            if index < i:
                continue

            # Go to next auction id if all the rows corresponding to current auction id are done being iterated
            if id != row['auctionid']:
                i = index
                break

            # Calculate and set data to append in result dataset
            max_bid = max(max_bid, row['enc_bid'])
            bidder = row['enc_bidder']
            item = row['item']
        
        # Append the encrypted max bid and encrypted bidder username of the winner along with auction id and product name in the result dataset
        result_dataset.at[idx, 'auctionid'] = id
        result_dataset.at[idx, 'enc_win_bid'] = max_bid
        result_dataset.at[idx, 'enc_win_bidder'] = bidder
        result_dataset.at[idx, 'item'] = item
        idx += 1

    # Save the result dataset in a csv which will be accessed by users to find out if they won in any auction
    result_dataset.to_csv("winners.csv")

    auc_id = result_dataset.at[0, 'auctionid']
    item = result_dataset.at[0, 'item']
    bidder = result_dataset.at[0, 'enc_win_bidder']

    # Winner and loser view of data when they decrypt values of any specific auction id with their private keys to find out if they won or not
    print("(Winner View)\nBidder", rsa_decrypt(bidder, kp_dec['eli.flint@flightsafety.co'][1], n), "won", item, "at auction", auc_id, "!!\n")
    print("(Loser View)\nBidder", rsa_decrypt(bidder, kp_dec['schadenfreud'][1], n), "won", item, "at auction", auc_id, "!!")

if __name__== '__main__':
    main()