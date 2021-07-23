import React from 'react';

const SuggestedVideo = ({ imgURL, VideoID, setCurrentVideoId, videoName, currentVideoIndexInPlaylist, setCurrentVideoIndexInPlaylist, videoPlaylist }) => {

    const handleClick = (event) => {
        setCurrentVideoId(event.target.id);
        for (var i = 0; i < videoPlaylist.length; i++) {
            if (event.target.id == videoPlaylist[i].id) {
                setCurrentVideoIndexInPlaylist(i);
                break;
            }
        }
    }

    return (
        <tr className="suggestedVideo">
            <td>
                <img id={VideoID} className="suggestedImg" src={imgURL} onClick={handleClick}></img>
            </td>
            <td>
                <button id={VideoID} className="link" onClick={handleClick}>{videoName}</button>
            </td>
        </tr>
    );
}

export default SuggestedVideo;