import React from "react";
import axios from "axios";
import Fire from "../../config/Fire.js"; // Firebase
import "./styles.scss"; // Styles
import Slide from "react-reveal";
import dog_wash from "../ServicesForm/cards"
import { service } from "firebase-functions/lib/providers/analytics";

class ServicesForm extends React.Component {
  
  constructor(props) {
    super(props);

    this.state = {
      latitude: 0,
      longitude: 0,
      results:"",
      errorStste:null,
      loading:false,
      city:"",
      apikey:"",
    };
    this.trackLocation=this.trackLocation.bind(this);
    this.getUpdates=this.getUpdates.bind(this);
  }

  // componentDidMount() {
  //   document.getElementById("coordinates").className = "collapse";
  // }

  updatePostion(position) {
    var latitude = position.coords.latitude;
    var longitude=position.coords.longitude;
    this.setState({latitude:latitude,longitude:longitude});
    document.getElementById("coordinates").className = "collapse.show";
  }

  trackLocation() {
    setInterval(() => {
      navigator.geolocation.getCurrentPosition((position) => {
        this.updatePostion(position);
      },
       (error) => {},
       {enableHighAccuracy: true, timeout: 20000, maximumAge: 10000}
      );
    });
    //this.getUpdates();
  }

 getUpdates() {
   // Get user's city
  var city = "";
  Fire.firestore().collection("parents").where("email", "==", Fire.auth().currentUser.email).get()
  .then(snapshot => {
    snapshot.forEach(doc => {
      city = doc.data().city;
      //document.getElementById("services-search").value += doc.data().city;
    })
  });
  Fire.firestore().collection("boarders").where("email", "==", Fire.auth().currentUser.email).get()
  .then(snapshot => {
    snapshot.forEach(doc => {
      city = doc.data().city;
      //document.getElementById("services-search").value += doc.data().city;
    })
  });
  Fire.firestore().collection("boarders").where("email", "==", Fire.auth().currentUser.email).get()
  .then(snapshot => {
    snapshot.forEach(doc => {
      city = doc.data().city;
      //document.getElementById("services-search").value += doc.data().city;
    })
  });

  setTimeout(() => {
    // Get user's search
    var search = document.getElementById("services-search").value.split(" ");
    var googlePlacesRef = "https://cors-anywhere.herokuapp.com/https://maps.googleapis.com/maps/api/place/textsearch/json?query=";
    for (var i=0; i<search.length; i++) { // for heroku
    if (i < search.length-1)
      googlePlacesRef += search[i]+"+";
    else
      googlePlacesRef += search[i]+" in "+city+"&key=";
  }
    var apiKey = "API-Key goes here";
    googlePlacesRef += apiKey;
    console.log(googlePlacesRef);
    var request = require('request');
    var options = { // 'https://cors-anywhere.herokuapp.com/https://maps.googleapis.com/maps/api/place/textsearch/json?query=restaurants+in+LongBeach&key=AIzaSyAr-Fd30VXDDoStitEIztHwyyYLVvjBIl4'
      'method': 'GET',
      'url': googlePlacesRef,
      'headers': {
      }
    };
    request(options, function (error, response) { 
      if (error) throw new Error(error);
      document.getElementById("results").innerHTML = null;
      var results = response.body.split('"formatted_address"');
      // Format results from Google so output is usable
      for (var i=1; i<results.length; i++) {
          var resultBox = document.createElement("div");
          resultBox.className = "service-box"
          var formattedResult = results[i];
          formattedResult = formattedResult.replace(/:/g,"");
          formattedResult = formattedResult.replace(/\[/g,"");
          formattedResult = formattedResult.replace(/\]/g,"");
          formattedResult = formattedResult.replace(/{/g,"");
          formattedResult = formattedResult.replace(/}/g,"");
          formattedResult = formattedResult.replace(/"/g,"");
          formattedResult = formattedResult.replace("geometry","");
          formattedResult = formattedResult.replace("location","\n");
          formattedResult = formattedResult.replace("southwest","");
          formattedResult = formattedResult.replace(/lat/g,"");
          formattedResult = formattedResult.replace(/lng/g,"");
          formattedResult = formattedResult.replace(/\n/g,"");
          while (formattedResult.includes("  "))
            formattedResult = formattedResult.replace(/  /," ");
          formattedResult = formattedResult.replace(/viewport.*name/,"");
          formattedResult = formattedResult.replace(/opening_hours.*rating /,"⭐");
          formattedResult = formattedResult.replace(/, reference.*user_ratings_total /," (");
          formattedResult = formattedResult.substring(0, formattedResult.length - 3);
          formattedResult += ")";
          var formattedResults = formattedResult.split(", ");
          var name = document.createElement("div");
          name.className = "box-row";
          name.textContent = formattedResults[6];
          resultBox.appendChild(name);
          var rating = document.createElement("div");
          rating.className = "box-row";
          rating.textContent = formattedResults[7];
          resultBox.appendChild(rating);
          var address = document.createElement("div");
          address.className = "box-row";
          address.textContent = formattedResults[0]+", "+formattedResults[1]+", "+formattedResults[2];
          resultBox.appendChild(address);
          // var location = document.createElement("div");
          // location.className = "box-row";
          // location.textContent = "("+formattedResults[4]+", "+formattedResults[5]+")";
          // resultBox.appendChild(location);
          document.getElementById("results").appendChild(resultBox);
      }
    });
   }, 1000);
 }

  render() {
    return (
      <div id="services-form">
        {/* <button className="mb-3" type="button" onClick={this.trackLocation}>Detect your location</button>
        <div id="coordinates">({this.state.latitude}, {this.state.longitude})</div> */}
        <div className="row my-3">
          <input className="search-bar mr-3" id="services-search" placeholder="Search..." maxLength="50" onChange={this.updateSearch} value={this.state.search}></input>
          <button onClick={this.getUpdates}>Search</button>
        </div>
        <div className="row" id="results"/>
        </div>
      
    );
  }
}

export default ServicesForm;