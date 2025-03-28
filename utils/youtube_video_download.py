import yt_dlp

url = "https://www.youtube.com/watch?v=Zn5jJwMS7Q4"

ydl_opts = {
    'format': 'bestvideo[height<=1080]+bestaudio/best',
    'outtmpl': '/Users/mingyu/Desktop/Fancam_Maker_V2/video_files/%(title)s.%(ext)s',
    'merge_output_format': 'mp4',  # 비디오와 오디오를 mp4 형식으로 병합
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])