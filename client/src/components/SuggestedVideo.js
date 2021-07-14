import React from 'react';

const SuggestedVideo = ({imgURL,VideoID,setCurrentVideoId,videoName})=>{
    
    const handleClick = (event) => {
        setCurrentVideoId(event.target.id);
        //ADD PLAYLIST INDEX UPDATE
    }
    return(
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