"""
1. 추적할 아이돌 이름 input 받기
2. 각종 class 객체 생성 & 기능 수행
3. 직캠 저장
"""

import yt_dlp

url = "https://www.youtube.com/watch?v=oRg-7iku6Ok"

ydl_opts = {
    'format': 'bestvideo[height<=1080]+bestaudio/best',
    'outtmpl': '%(title)s.%(ext)s',
    'merge_output_format': 'mp4',  # 비디오와 오디오를 mp4 형식으로 병합
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

#
# import yt_dlp
#
# url = "https://www.youtube.com/watch?v=UpEPkPg8YP4"
#
# ydl_opts = {
#     'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',  # MP4 형식의 최적 화질 선택
#     'outtmpl': '%(title)s.%(ext)s',
#     'postprocessors': [{
#         'key': 'FFmpegVideoConvertor',
#         'preferedformat': 'mp4',  # mp4로 변환
#     }],
# }
#
# with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#     ydl.download([url])
