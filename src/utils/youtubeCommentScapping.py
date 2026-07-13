from googleapiclient.discovery import build
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY=os.getenv("YOUTUBE")

def get_video_id(url):
    return url.split("v=")[-1]

def get_comments(video_url, max_comments=2000):
    video_id = get_video_id(video_url)

    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
    comments = []
    next_page_token = None
    temp=None
    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            temp=snippet
            comments.append({
                "CommentText": snippet["textDisplay"],
                "Likes": snippet["likeCount"],
                "PublishedAt": snippet["publishedAt"]
            })

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break
    print(temp)
    return pd.DataFrame(comments)


if __name__=="__main__":
    print(get_comments("https://www.youtube.com/watch?v=Cb6wuzOurPc"))