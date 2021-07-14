import React, {useEffect,useState} from 'react';



const Description =({currentVideoId,videoDescriptionVisual, setVideoDescriptionVisual, firstVideoURL, currVideoDescription, setCurrVideoDescription}) =>{

    useEffect(()=>{
        const getVideoDescription = async () => {
            try {             
                const res = await fetch(
                   "https://youtube.googleapis.com/youtube/v3/videos?part=snippet&id="+currentVideoId+"&fields=items.snippet(description)&key=AIzaSyCuVFHDltJZeTbesYt0J2eodWwwfqkpELA",
                    { method: "GET"}
                );
                //console.log(res); // logs the response from the YouTube API - i.e status code, response headers...
                const data = await res.json(); // turns the reponse data to a json object
                setCurrVideoDescription(data.items[0].snippet.description);
                setVideoDescriptionVisual("videoDescriptionTextArea");
            } catch (err) {
                console.log(err);
            }
        }
        getVideoDescription();
    },[currentVideoId])

    return(
        
        <div className="videoDescription">
            <label id={videoDescriptionVisual}>{currVideoDescription}</label>
        </div>
    );
}

export default Description;