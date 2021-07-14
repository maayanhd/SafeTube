import React, {useEffect,useState} from 'react';
import ReactPlayer from 'react-player/youtube'

import SuggestedVideo from './SuggestedVideo';
import VideoInPlaylist from './VideoInPlaylist';



const PlayerAndPlaylist =({currentVideoId, setCurrentVideoId,currentVideoIndexInPlaylist,
    setCurrentVideoIndexInPlaylist,videoPlaylist, playBoxVisual,suggestedPlaylistVideos,
    setSuggestedPlaylistVideos, suggestedVideosVisual}) =>{
    
    let suggestedVids = [];

    const incVideoIndex=()=>{
        setCurrentVideoIndexInPlaylist(currentVideoIndexInPlaylist+1);
    }
    
    useEffect(()=>{
        console.log("hi"); 
        if(videoPlaylist.length > currentVideoIndexInPlaylist){   
                
            setCurrentVideoId(videoPlaylist[currentVideoIndexInPlaylist].id);
        }
    },[videoPlaylist,currentVideoIndexInPlaylist])

    useEffect(()=>{
        console.log("hi2");
        if(videoPlaylist.length > 1){         
            for(var i = 0; i < videoPlaylist.length; i++){
                suggestedVids.push(<SuggestedVideo 
                     imgURL={videoPlaylist[i].imgURL} VideoID = {videoPlaylist[i].id} 
                     setCurrentVideoId={setCurrentVideoId} videoName={videoPlaylist[i].videoName}
                     />
                );
            }
            setSuggestedPlaylistVideos(suggestedVids);
        }
    },[videoPlaylist])

    return(
            
            <div className={playBoxVisual}> 
                <ReactPlayer
                url={"https://www.youtube.com/embed/"+currentVideoId}
                controls={true}
                onEnded={incVideoIndex}
                />
                <table id={suggestedVideosVisual}>
                  <tr>
                      <th>Playlist videos</th>
                      <th></th>
                  </tr>

                      {suggestedPlaylistVideos.map((r) => (
                          
                              <tr>{r}</tr>
                          
                        ))}     
                  
                </table>
            </div>
    );
    
}

export default PlayerAndPlaylist;