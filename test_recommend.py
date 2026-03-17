from dataset.music_player_integration import MusicPlayerIntegration

player = MusicPlayerIntegration()

# Play a song
player.play_song('Afterglow', 'Ed Sheeran', '0')

# Like a song
player.like_song('Afterglow', 'Ed Sheeran', '0')

# Dislike a song
player.dislike_song('Missing Piece', 'Vance Joy', '0')

# Get advanced recommendations for genre '0'
recs = player.get_advanced_recommendations('0', n=5)
print('[TEST] Advanced recommendations:', recs)

