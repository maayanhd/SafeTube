import React, { useEffect, useState } from 'react';
import ReactPlayer from 'react-player/youtube'
import ChannelDetails from './ChannelDetails';

import SuggestedVideo from './SuggestedVideo';



const PlayerAndPlaylist = ({ currentVideoId, setCurrentVideoId, currentVideoIndexInPlaylist,
    setCurrentVideoIndexInPlaylist, videoPlaylist, playBoxVisual, suggestedPlaylistVideos,
    setSuggestedPlaylistVideos, suggestedVideosVisual, channelDetailsDisplay }) => {

    let suggestedVids = [];

    const incVideoIndex = () => {
        setCurrentVideoIndexInPlaylist(currentVideoIndexInPlaylist + 1);
    }

    useEffect(() => {

        if (videoPlaylist.length > currentVideoIndexInPlaylist) {

            setCurrentVideoId(videoPlaylist[currentVideoIndexInPlaylist].id);
        }
    }, [videoPlaylist, currentVideoIndexInPlaylist])

    useEffect(() => {
        if (videoPlaylist.length > 1) {
            for (var i = 0; i < videoPlaylist.length; i++) {
                suggestedVids.push(<SuggestedVideo
                    imgURL={videoPlaylist[i].imgURL} VideoID={videoPlaylist[i].id}
                    setCurrentVideoId={setCurrentVideoId} videoName={videoPlaylist[i].videoName} currentVideoIndexInPlaylist={currentVideoIndexInPlaylist}
                    setCurrentVideoIndexInPlaylist={setCurrentVideoIndexInPlaylist} videoPlaylist={videoPlaylist}
                />
                );
            }
            setSuggestedPlaylistVideos(suggestedVids);
        }
    }, [videoPlaylist])

    return (

        <div className={playBoxVisual}>
            <ChannelDetails videoPlaylist={videoPlaylist} currentVideoIndexInPlaylist={currentVideoIndexInPlaylist} channelDetailsDisplay={channelDetailsDisplay} />
            <ReactPlayer
                url={"https://www.youtube.com/embed/" + currentVideoId}
                controls={true}
                onEnded={incVideoIndex}
                width="700px"
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