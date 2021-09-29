import React from 'react';

import { useEffect, useState } from 'react';

import CircularProgress from '@material-ui/core/CircularProgress';

const KEY = 'AIzaSyCuVFHDltJZeTbesYt0J2eodWwwfqkpELA';

const resultsFromPlaylist = "50";

let currentPlaylist = [];


class video {
    constructor(videoName, channelName, imgURL, rating, id, like, dislike , views, chBadVidAmount,chGoodVidAmount) {
        this.videoName = videoName;
        this.channelName = channelName;
        this.imgURL = imgURL;
        this.rating = rating;
        this.id = id;
        this.like = like;
        this.dislike = dislike;
        this.views = views;
        this.chBadVidAmount = chBadVidAmount;
        this.chGoodVidAmount = chGoodVidAmount;
    }
    getName() {
        return (this.videoName);
    }
}

const Form = ({ setCurrentVideoIndexInPlaylist, getIDFromURL, setVideoPlaylist, setPlayBoxVisual, setInputText, inputText, firstVideoURL
    , setFirstVideoURL, setSuggestedVideosVisual, setChannelDetailsDisplay, setvideoDetailsDisplay, loading, setLoading, validURL, setValidURL}) => {

    const [currentPlaylistID, setCurrentPlaylistID] = useState("");
    const [errMSG,setErrMSG] = useState("");

    let startOfPlayListIDIndex = 0;
    let endOfPlayListID = 0;
    

        const hidePlayer=()=>{
            setPlayBoxVisual("hiddenPlayerBox");
            setvideoDetailsDisplay("videoDetailsHidden");
        }

    const inputTextHandler = (e) => {
        setInputText(e.target.value);
    };
    const placeSingleVideoIntoPlaylist = async () => {
        try {
            setLoading(false);
            const res = await fetch(
                "http://localhost:8082/api/v1/youtube/video?videoid=" + getIDFromURL(firstVideoURL) ,
                
                { method: "GET"  }
            );
            const data = await res.json(); // turns the reponse data to a json object
            setLoading(true);
            setChannelDetailsDisplay("channelDetails");
            console.log(data);
            currentPlaylist = [];
            setVideoPlaylist([]);
            if(data.scoring.final_score == undefined){
                data.scoring = data.scoring[0];
            }
            currentPlaylist.push(new video(data.title, data.uploader, 
            data.thumbnail, data.scoring.final_score,
            data.display_id, data.like_count,
            data.dislike_count, data.view_count,
            data.channelScore.badVideoCount,data.channelScore.goodVideoCount));
            console.log(currentPlaylist);
            setVideoPlaylist(currentPlaylist);
        } catch (err) {
            console.log(err);
            setErrMSG("Something went wrong.");  
            setValidURL("validURLSeen");
            hidePlayer();
        }
    }


    useEffect(() => {
        if (isURLValid()) {
            setPlayBoxVisual("playerBox");
            setvideoDetailsDisplay("videoDetails");

            //this finds the start and end of playlist ID
            startOfPlayListIDIndex = firstVideoURL.indexOf('list=') + 5;
            endOfPlayListID = firstVideoURL.length;
            setCurrentPlaylistID("");
            if (firstVideoURL.includes("&", startOfPlayListIDIndex)) {
                endOfPlayListID = firstVideoURL.indexOf("&", startOfPlayListIDIndex);
            }
            if (firstVideoURL.includes("list=")) {
                setCurrentPlaylistID(firstVideoURL.substring(startOfPlayListIDIndex, endOfPlayListID));
                
            }
            else {
                placeSingleVideoIntoPlaylist();
                setSuggestedVideosVisual("hiddenPlaylistTable");
                
            }
        } else {
            hidePlayer();
        }
    }, [firstVideoURL]);


    const searchButtonHandler = (e) => {
        e.preventDefault();
            let defaultURLString = inputText.toString();
            defaultURLString = defaultURLString.replace('watch?v=', 'embed/');
            if (defaultURLString.length !== 0) {
                setFirstVideoURL(defaultURLString);
                //this sets the currect video in playlist ( if user didnt start at the first video)
                if (defaultURLString.includes("index=")) {
                    let startVidNumIndex = defaultURLString.indexOf("index=") + 6;
                    let endVidNumIndex = defaultURLString.length;
                    if (defaultURLString.substring(startVidNumIndex, endVidNumIndex).includes("&")) {
                        endVidNumIndex = defaultURLString.indexOf("&", startVidNumIndex);
                    }
                    let idx = defaultURLString.substring(startVidNumIndex, endVidNumIndex) - "1";
                    if(idx<"50"){
                        setCurrentVideoIndexInPlaylist(idx);
                    }else{
                        setCurrentVideoIndexInPlaylist(0);
                    }

                } else {
                    setCurrentVideoIndexInPlaylist(0);
                }
                setInputText("");
            }
        
    }

    useEffect(() => {
        const handleSubmit = async () => {
            setLoading(false);
            try {
                const res = await fetch("http://localhost:8082/api/v1/youtube/playlist?playlistid=" +currentPlaylistID, {
                    method: "GET",
                    headers: {
                        "Content-Type": "application/json",
                        },               
                });
                const data = await res.json();
                setLoading(true);
                setChannelDetailsDisplay("channelDetails");
                setvideoDetailsDisplay("videoDetails");
                setSuggestedVideosVisual("playlistTable");
                
                console.log(data);
            
                currentPlaylist=[];
                for (var i = 0; i < Math.min(resultsFromPlaylist - "0", data.length); i++) {
                    if(data.title !=="Private video"){
                        if(data[i] != null){
                            if(data[i].scoring.final_score == undefined){
                                data[i].scoring = data[i].scoring[0];
                            }
                            let vid = new video(data[i].title, data[i].uploader, 
                                data[i].thumbnail, data[i].scoring.final_score,
                                data[i].display_id, data[i].like_count,
                                data[i].dislike_count, data[i].view_count,
                                data[i].channelScore.badVideoCount,data[i].channelScore.goodVideoCount);
                            console.log(vid);
                            console.log(i);
                            
                            currentPlaylist.push(vid);
                        }
                        
                        
                    }
                }
                setVideoPlaylist(currentPlaylist);
            } catch (error) {
                console.log(error);
                setValidURL("validURLSeen");
                setErrMSG("Something went wrong.");  
                hidePlayer();
            }
        }
        if(currentPlaylistID!=""){
            handleSubmit();
        }else{
            setLoading(true);
        }
        
        
    }, [currentPlaylistID]);

    const isURLValid = () => {
        console.log(firstVideoURL);
        let valid = false;
        if(firstVideoURL){
            valid = firstVideoURL.startsWith('https://www.youtube.com/watch?v=') || firstVideoURL.startsWith('https://www.youtube.com/embed/') ;
            console.log(valid);
            if(valid){
                setValidURL("validURLHidden");
            }else{
                setValidURL("validURLSeen");
                setErrMSG("URL not valid.");                
            }
        }
        return valid;
    }
    
    return (
        <div>
            <form>
                <input value={inputText} onChange={inputTextHandler} type="text" className="playlist-input" />
                <button onClick={searchButtonHandler} className="playlist-url-button" type="submit">
                    <i className="fas fa-search"></i>
                </button>
                {loading ? (""):(<CircularProgress />)}
            </form>
            <div id={validURL}>{errMSG}</div>
        </div>
    );

}



export default Form;