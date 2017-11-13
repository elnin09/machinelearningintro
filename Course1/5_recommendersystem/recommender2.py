import graphlab

song_data = graphlab.SFrame('song_data.gl/')

song_data['song'].show()

kanye_west = song_data[song_data['artist'] == 'Kanye West']['user_id'].unique()
len(kanye_west)

ff = song_data[song_data['artist'] == 'Foo Fighters']['user_id'].unique()
len(ff)

ts = song_data[song_data['artist'] == 'Taylor Swift']['user_id'].unique()
len(ts)

lg = song_data[song_data['artist'] == 'Lady GaGa']['user_id'].unique()
len(lg)
