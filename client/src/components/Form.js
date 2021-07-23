import React from 'react';

import { useEffect, useState } from 'react';

const KEY = 'AIzaSyCuVFHDltJZeTbesYt0J2eodWwwfqkpELA';

const resultsFromPlaylist = "10";

let currentPlaylist=[];

class video{
    constructor(videoName,description,channelName,imgURL,rating,id){
        this.videoName=videoName;
        this.description=description;
        this.channelName=channelName;
        this.imgURL=imgURL;
        this.rating=rating;
        this.id=id;
    }
    getName(){
        return(this.videoName);
    }
}
 
const Form =({setCurrentVideoIndexInPlaylist,getIDFromURL, setVideoPlaylist, setPlayBoxVisual,setInputText,inputText, firstVideoURL, setFirstVideoURL, setSuggestedVideosVisual}) => {
   
    const [currentPlaylistID, setCurrentPlaylistID]= useState("");

    let startOfPlayListIDIndex = 0;
    let endOfPlayListID = 0;
    let privateVideoAmount;

    const inputTextHandler = (e) =>{
        setInputText(e.target.value);
    };
    const placeSingleVideoIntoPlaylist = async () => {
        try {             
            const res = await fetch(
               "https://youtube.googleapis.com/youtube/v3/videos?id="+getIDFromURL(firstVideoURL)+"&part=snippet&fields=items%2Fsnippet(title%2Cdescription%2CchannelTitle)&key=AIzaSyD5aoCjGplRER2cPriT28Osh7McTSW6QDk",
                { method: "GET"}
            );
            const data = await res.json(); // turns the reponse data to a json object
            currentPlaylist=[];
            setVideoPlaylist(currentPlaylist);
            currentPlaylist.push(new video(data.items[0].snippet.title,data.items[0].snippet.description,data.items[0].snippet.channelTitle,"none","9",getIDFromURL(firstVideoURL)));
            setVideoPlaylist(currentPlaylist);
            console.log(currentPlaylist[0].getName());
        } catch (err) {
            console.log(err);
        }
    } 

    useEffect(()=>{
        if(validURL()){
            
            setPlayBoxVisual("playerBox");
            //this finds the start and end of playlist ID
            startOfPlayListIDIndex = firstVideoURL.indexOf('list=')+5;
            endOfPlayListID = firstVideoURL.length;
            if(firstVideoURL.includes("&",startOfPlayListIDIndex)){
                endOfPlayListID = firstVideoURL.indexOf("&",startOfPlayListIDIndex);
            }
            if(firstVideoURL.includes("list=")){
                setCurrentPlaylistID("");
                setCurrentPlaylistID(firstVideoURL.substring(startOfPlayListIDIndex,endOfPlayListID));
                setSuggestedVideosVisual("playlistTable");
            }
            else{
                placeSingleVideoIntoPlaylist();
                setSuggestedVideosVisual("hiddenPlaylistTable");
            }
        }else{
            setPlayBoxVisual("hiddenPlayerBox");
        }        
    },[firstVideoURL]);
       

    const searchButtonHandler = (e)=>{
        e.preventDefault();
        let defaultURLString = inputText.toString();
        defaultURLString = defaultURLString.replace('watch?v=', 'embed/');
        if(defaultURLString.length !== 0){
            setFirstVideoURL(defaultURLString);
            //this sets the currect video in playlist ( if user didnt start at the first video)
            if(defaultURLString.includes("index=")){
                let startVidNumIndex = defaultURLString.indexOf("index=")+6;
                let endVidNumIndex = defaultURLString.length;
                if(defaultURLString.substring(startVidNumIndex,endVidNumIndex).includes("&")){
                    endVidNumIndex = defaultURLString.indexOf("&",startVidNumIndex);
                }
                setCurrentVideoIndexInPlaylist(defaultURLString.substring(startVidNumIndex,endVidNumIndex)-"1");
            }else{
                setCurrentVideoIndexInPlaylist(0);
            }
            setInputText("");
        }
    }
    
    useEffect(()=>{
            const handleSubmit = async () => {
            try {             
                const res = await fetch(
                    "https://www.googleapis.com/youtube/v3/playlistItems?maxResults="+resultsFromPlaylist+"&playlistId="+currentPlaylistID+"&part=snippet&fields=items%2Fid%2C%20items%2Fsnippet(title%2Cdescription%2CchannelTitle%2Cthumbnails(medium)%2CresourceId)&key=AIzaSyD5aoCjGplRER2cPriT28Osh7McTSW6QDk",
                    { method: "GET"}
                );
                const data = await res.json(); // turns the reponse data to a json object
                privateVideoAmount = 0;
                for(var i=0;i<data.items.length;i++){
                    if(data.items[i].snippet.title==="Private video"){
                        privateVideoAmount++;
                    }
                }
                currentPlaylist=[];
                setVideoPlaylist(currentPlaylist);
                for(var i = 0; i < Math.min(resultsFromPlaylist - "0",data.items.length-privateVideoAmount); i++){
                    let vid = new video(data.items[i].snippet.title,data.items[i].snippet.description,data.items[i].snippet.channelTitle,data.items[i].snippet.thumbnails.medium.url,"9",data.items[i].snippet.resourceId.videoId);
                    currentPlaylist.push(vid);
                }
                setVideoPlaylist(currentPlaylist);
            } catch (err) {
                console.log(err);
            }
        }
        handleSubmit();
    },[currentPlaylistID]);

    const validURL =()=>{
        return true;
    }


    return(
        <div>
            <form>
                <input value = {inputText} onChange={inputTextHandler} type="text" className="playlist-input" />
                <button  onClick={searchButtonHandler} className="playlist-url-button" type="submit">
                <i className="fas fa-search"></i>
                </button>
            </form>
      </div>
    );

}



export default Form;