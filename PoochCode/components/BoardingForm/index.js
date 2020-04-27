import React from "react";
import Fire from "../../config/Fire.js"; // Firebase
import "./styles.scss"; // Styles
// Animation
import Loader from "react-loader-spinner";
import "../BoardingForm/styles.scss";
import Close from "../MeetupForm/close.jpg";

class BoardingForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      name: "",
      phone: "",
      dailyRate: "",
      address: "",
      city: "",
      search: "",
      reset: false
    };
    this.handleChange = this.handleChange.bind(this);
    this.updateSearch = this.updateSearch.bind(this);
    this.fillStar = this.fillStar.bind(this);
  }

  handleChange(e) {
    this.setState({ [e.target.name]: e.target.value });
  }

  updateSearch(e) {
    this.setState({search: e.target.value.substr(0,50) });
  }

  componentDidMount() {
    document.getElementById("loader").style.display = "block";
    setTimeout(() => {
      document.getElementById("loader").style.display = "none";
    }, 1750);
  }

  componentDidUpdate() {
    document.getElementById("loader").style.display = "block";
    document.getElementById("bubble-home").innerHTML = null; // Clear old data before updating
    // Order results based on search criteria
    Fire.firestore().collection("boarders")
    .orderBy(document.getElementById("boarder-search-category").value).get()
    .then(snapshot => {
      snapshot.docs.forEach(doc => {
        renderBoarders(doc);
      });
    });

    function renderBoarders(doc) {
      // Bubble render
      var newBox = document.createElement("div");
      newBox.className = "boarder-box";
      newBox.addEventListener("click", ()=>{viewProfile(doc.data().email)});
      var picHolder = document.createElement("div");
      picHolder.className = "mb-3";
      var newPic = document.createElement("img");
      newPic.className = "box-pic";
      newPic.src = doc.data().pic;
      newPic.alt = "Profile Picture";
      picHolder.appendChild(newPic);
      newBox.appendChild(picHolder);
      var newCol = document.createElement("div");
      newCol.className = "col";
      newBox.appendChild(newCol);

      var ratingRow = document.createElement("div"); // First row
      ratingRow.className = "my-2";
      newCol.appendChild(ratingRow);
      var rating = document.createElement("div");
      rating.className = "box-rating";
      var ratingSum = 0;
      var ratingCount = 0;
      Fire.firestore().collection("reviews")
      .where("for", "==", doc.data().email)
      .get()
      .then(snapshot => {
        snapshot.docs.forEach(doc => {
          ratingSum += doc.data().stars;
          ratingCount++;
        });
        if (ratingCount > 0)
          rating.textContent = (ratingSum/ratingCount).toFixed(1)+" ⭐ ("+ratingCount+")";
        else
          rating.textContent = "No Ratings ⭐";
      });
      ratingRow.appendChild(rating);

      var nameRow = document.createElement("div"); // Second row
      nameRow.className = "mx-1 box-row";
      newCol.appendChild(nameRow);
      var name = document.createElement("div");
      // Make name camelcase
      var nameWords = doc.data().name.toLowerCase().split(" ");
      var adjustedName = "";
      for (var i=0; i<nameWords.length; i++)
        adjustedName += nameWords[i].substr(0,1).toUpperCase() + nameWords[i].substr(1, nameWords[i].length-1) + " ";
      name.textContent = adjustedName;
      name.className = "boarder-name";
      nameRow.appendChild(name);

      var phoneRow = document.createElement("div"); // Third row
      phoneRow.className = "mx-1 box-row";
      newCol.appendChild(phoneRow);
      var phone = document.createElement("div");
      phone.textContent = doc.data().phone;
      if (phone.textContent.length === 10)
        phone.textContent = phone.textContent[0]+phone.textContent[1]+phone.textContent[2]+"-"+phone.textContent[3]+phone.textContent[4]+phone.textContent[5]+"-"+phone.textContent[6]+phone.textContent[7]+phone.textContent[8]+phone.textContent[9];
      if (phone.textContent.length === 7)
        phone.textContent = phone.textContent[0]+phone.textContent[1]+phone.textContent[2]+"-"+phone.textContent[3]+phone.textContent[4]+phone.textContent[5]+phone.textContent[6];
      phone.className = "boarder-phone";
      phoneRow.appendChild(phone);

      var addressRow = document.createElement("div"); // Fourth row
      addressRow.className = "mx-1 box-row";
      newCol.appendChild(addressRow);
      var address = document.createElement("div");
      // Make address camelcase
      var addressWords = doc.data().address.toLowerCase().split(" ");
      var adjustedAddress = "";
      for (var i=0; i<addressWords.length; i++)
        adjustedAddress += addressWords[i].substr(0,1).toUpperCase() + addressWords[i].substr(1, addressWords[i].length-1) + " ";
      address.textContent = adjustedAddress;
      address.className = "boarderer-address";
      addressRow.appendChild(address);

      var cityRow = document.createElement("div"); // Fifth row
      cityRow.className = "mx-1 box-row";
      newCol.appendChild(cityRow);
      var city = document.createElement("div");
      // Make city camelcase
      var cityWords = doc.data().city.toLowerCase().split(" ");
      var adjustedCity = "";
      for (var i=0; i<cityWords.length; i++)
        adjustedCity += cityWords[i].substr(0,1).toUpperCase() + cityWords[i].substr(1, cityWords[i].length-1) + " ";
      city.textContent = adjustedCity;
      city.className = "boarder-city";
      cityRow.appendChild(city);

      var dailyRateRow = document.createElement("div"); // Sixth row
      dailyRateRow.className = "mx-1 box-row";
      newCol.appendChild(dailyRateRow);
      var dailyRate = document.createElement("div");
      dailyRate.textContent = "$"+doc.data().dailyRate+"/day";
      dailyRate.className = "boarder-daily-rate";
      dailyRateRow.appendChild(dailyRate);


      // Update popup rating
      if (document.getElementById("popup-boarder-email").textContent === doc.data().email) {
        Fire.firestore().collection("reviews")
        .where("for", "==", document.getElementById("popup-boarder-email").textContent)
        .get()
        .then(snapshot => {
        var ratingSum = 0;
        var ratingCount = 0;
          snapshot.docs.forEach(doc => {
            ratingSum += doc.data().stars;
            ratingCount++;
          });
          if (ratingCount > 0)
            document.getElementById("popup-boarder-rating").textContent = (ratingSum/ratingCount).toFixed(1)+" ⭐ ("+ratingCount+")";
          else
            document.getElementById("popup-boarder-rating").textContent = "No Ratings ⭐";
        });
      }

      // Search functionality
      var matchSearch = true;
      // Searching by address
      if (document.getElementById("boarder-search-category").value === "address") { 
        if (document.getElementById("boarder-search").value.length > address.textContent.length) // Check if search bar is longer than address
          matchSearch = false;
        else
          for (var j=0; j<document.getElementById("boarder-search").value.length; j++) // Loop through each letter in search bar
            if (document.getElementById("boarder-search").value[j].toLowerCase() !== address.textContent[j].toLowerCase())
              matchSearch = false;
      }
      // Searching by city
      if (document.getElementById("boarder-search-category").value === "city") { 
        if (document.getElementById("boarder-search").value.length > city.textContent.length) // Check if search bar is longer than city
          matchSearch = false;
        else
          for (var j=0; j<document.getElementById("boarder-search").value.length; j++) // Loop through each letter in search bar
            if (document.getElementById("boarder-search").value[j].toLowerCase() !== city.textContent[j].toLowerCase())
              matchSearch = false;
      }
      // Searching by daily rate
      if (document.getElementById("boarder-search-category").value === "dailyRate") { 
        if (document.getElementById("boarder-search").value.length > dailyRate.textContent.length) // Check if search bar is longer than daily rate
          matchSearch = false;
        else
          for (var j=0; j<document.getElementById("boarder-search").value.length; j++) // Loop through each letter in search bar
            if (document.getElementById("boarder-search").value[j].toLowerCase() !== dailyRate.textContent.substr(1,3)[j].toLowerCase())
              matchSearch = false;
      }
      // Searching by name
      if (document.getElementById("boarder-search-category").value === "name") { 
        if (document.getElementById("boarder-search").value.length > name.textContent.length) // Check if search bar is longer than name
          matchSearch = false;
        else
          for (var j=0; j<document.getElementById("boarder-search").value.length; j++) // Loop through each letter in search bar
            if (document.getElementById("boarder-search").value[j].toLowerCase() !== name.textContent[j].toLowerCase())
              matchSearch = false;
      }
      // Only add bubble if it matches search
      if (matchSearch === true)
        document.getElementById("bubble-home").appendChild(newBox);
      document.getElementById("loader").style.display = "none";  
    }

    function viewProfile(email) {
      document.getElementById("boarder-popup").className = "row collapse";
        Fire.firestore().collection("boarders")
        .where("email", "==", email)
        .get()
        .then(snapshot => {
          snapshot.docs.forEach(doc => {
            document.getElementById("popup-boarder-pic").src = doc.data().pic;
            document.getElementById("popup-boarder-email").textContent = doc.data().email;
            document.getElementById("popup-boarder-name").textContent = doc.data().name;
            if (doc.data().phone.length == 10)
            document.getElementById("popup-boarder-phone").textContent = doc.data().phone[0]+doc.data().phone[1]+doc.data().phone[2]+"-"+doc.data().phone[3]+doc.data().phone[4]+doc.data().phone[5]+"-"+doc.data().phone[6]+doc.data().phone[7]+doc.data().phone[8]+doc.data().phone[9];
            else
              document.getElementById("popup-boarder-phone").textContent = doc.data().phone;
            document.getElementById("popup-boarder-city").textContent = doc.data().city;
            document.getElementById("popup-boarder-daily-rate").textContent = "$"+doc.data().dailyRate+"/day";
            document.getElementById("popup-boarder-feature1-pic").src = doc.data().feature1;
            document.getElementById("popup-boarder-feature2-pic").src = doc.data().feature2;
            document.getElementById("popup-boarder-feature3-pic").src = doc.data().feature3;
            document.getElementById("popup-boarder-feature4-pic").src = doc.data().feature4;
            var pic1 = document.getElementById("popup-boarder-feature1-pic").src;
            if (pic1.substr(pic1.length-9, pic1.length-1) !== "undefined")
              document.getElementById("popup-boarder-feature1-pic").className = "popup-feature collapse.show";
            else
              document.getElementById("popup-boarder-feature1-pic").className = "popup-feature collapse";
            var pic2 = document.getElementById("popup-boarder-feature2-pic").src;
            if (pic2.substr(pic2.length-9, pic2.length-1) !== "undefined")
                document.getElementById("popup-boarder-feature2-pic").className = "popup-feature collapse.show";
            else
                document.getElementById("popup-boarder-feature2-pic").className = "popup-feature collapse";
            var pic3 = document.getElementById("popup-boarder-feature3-pic").src;
            if (pic3.substr(pic3.length-9, pic3.length-1) !== "undefined")
              document.getElementById("popup-boarder-feature3-pic").className = "popup-feature collapse.show";
            else
              document.getElementById("popup-boarder-feature3-pic").className = "popup-feature collapse";
            var pic4 = document.getElementById("popup-boarder-feature4-pic").src;
            if (pic4.substr(pic4.length-9, pic4.length-1) !== "undefined")
              document.getElementById("popup-boarder-feature4-pic").className = "popup-feature collapse.show";
            else
              document.getElementById("popup-boarder-feature4-pic").className = "popup-feature collapse";
            
          
        // Update popup average rating
        if (document.getElementById("popup-boarder-email").textContent === doc.data().email) {
          Fire.firestore().collection("reviews")
          .where("for", "==", document.getElementById("popup-boarder-email").textContent)
          .get()
          .then(snapshot => {
            var ratingSum = 0;
          var ratingCount = 0;
            snapshot.docs.forEach(doc => {
              ratingSum += doc.data().stars;
              ratingCount++;
            });
            if (ratingCount > 0)
              document.getElementById("popup-boarder-rating").textContent = (ratingSum/ratingCount).toFixed(1)+" ⭐ ("+ratingCount+")";
            else
              document.getElementById("popup-boarder-rating").textContent = "No Ratings ⭐";
          });
        }

        // Update popup user rating
        for (var i=0; i<5; i++) {
          document.getElementById("star"+(i+1)).className = "empty-stars";
          document.getElementById("star"+(i+1)).textContent = "☆";
        }
        Fire.firestore().collection("reviews")
        .where("by", "==", Fire.auth().currentUser.email)
        .where("for", "==", document.getElementById("popup-boarder-email").textContent)
        .get()
        .then(snapshot => {
          snapshot.docs.forEach(doc => {
            for (var i=0; i<5; i++) {
              if (i < doc.data().stars) {
                document.getElementById("star"+(i+1)).className = "full-stars";
                document.getElementById("star"+(i+1)).textContent = "⭐";
              }
              else {
                document.getElementById("star"+(i+1)).className = "empty-stars";
                document.getElementById("star"+(i+1)).textContent = "☆";
              }
            }
          })
        });

        // Get popup user review
        document.getElementById("new-review").value = "";
        document.getElementById("submit-review").textContent = "Submit Review";
        Fire.firestore().collection("reviews")
        .where("by", "==", Fire.auth().currentUser.email)
        .where("for", "==", document.getElementById("popup-boarder-email").textContent)
        .get()
        .then(snapshot => {
          snapshot.docs.forEach(doc => {
                document.getElementById("new-review").value = doc.data().body;
                document.getElementById("submit-review").textContent = "Update Review";
          })
        });

        // Get all popup reviews by others
        document.getElementById("review-col").innerHTML = null;
        Fire.firestore().collection("reviews")
        .where("for", "==", document.getElementById("popup-boarder-email").textContent)
        .get()
        .then(snapshot => {
          snapshot.docs.forEach(doc => {
                var reviewDiv = document.createElement("div");
                reviewDiv.className = "user-review my-4";
                var ratingDiv = document.createElement("div");
                ratingDiv.className = "row my-2";
                var nameDiv = document.createElement("div");
                Fire.firestore().collection("parents").where("email", "==", doc.data().by).get()
                .then(snapshot => {
                  snapshot.forEach(doc => {
                    nameDiv.textContent = doc.data().name;
                  })
                });
                Fire.firestore().collection("boarders").where("email", "==", doc.data().by).get()
                .then(snapshot => {
                  snapshot.forEach(doc => {
                    nameDiv.textContent = doc.data().name;
                  })
                });
                Fire.firestore().collection("boarders").where("email", "==", doc.data().by).get()
                .then(snapshot => {
                  snapshot.forEach(doc => {
                    nameDiv.textContent = doc.data().name;
                  })
                });
                ratingDiv.appendChild(nameDiv);
                for (var i=0; i<5; i++) {
                  var starDiv = document.createElement("div");
                  if (doc.data().stars > i) {
                    starDiv.textContent = "⭐";
                    //starDiv.className = "full-stars";
                  }
                  else {
                    starDiv.textContent = "☆";
                    starDiv.className = "empty-stars";
                  }
                  ratingDiv.appendChild(starDiv);
                }
                reviewDiv.appendChild(ratingDiv);
                var bodyDiv = document.createElement("div");
                bodyDiv.className = "review-body";
                bodyDiv.textContent = doc.data().body;
                reviewDiv.appendChild(bodyDiv);
                document.getElementById("review-col").appendChild(reviewDiv);
            })
        });

        });
        });
        setTimeout(() => {
          document.getElementById("boarder-popup").className = "fixed-top row collapse.show";
        }, 500);
    }
  }

  closeProfile() { document.getElementById("boarder-popup").className = "fixed-top row collapse"; }

  fillStar(event) {
    var rating = 0;
    for (var i=0; i<5; i++) {
      if(i < parseInt(event.target.id.split("star")[1]) ) {
        document.getElementById("star"+(i+1)).className = "full-stars";
        document.getElementById("star"+(i+1)).textContent = "⭐";
        rating++;
      }
      else {
        document.getElementById("star"+(i+1)).className = "empty-stars";
        document.getElementById("star"+(i+1)).textContent = "☆";
      }
    }
  }

  submitReview() {
    // Check if user selected a rating
    var rating = 0;
    for (var i=1; i<6; i++)
      if(document.getElementById("star"+i).className === "full-stars" )
        rating++;
    if (rating > 0) {
      if (document.getElementById("new-review").value.length > 0) {
        // Delete existing review
        Fire.firestore().collection("reviews")
        .where('by', '==', Fire.auth().currentUser.email)
        .where("for", "==", document.getElementById("popup-boarder-email").textContent)
        .get().then(snapshot => {
          snapshot.docs.forEach(doc => {
            doc.ref.delete();
          });
        });
        // Wait 1 second
        setTimeout(() => {
          // Add new review
          Fire.firestore().collection("reviews").add({
            by: Fire.auth().currentUser.email,
            for: document.getElementById("popup-boarder-email").textContent,
            body: document.getElementById("new-review").value,
            stars: rating
          });
          //this.setState({reset: !this.state.reset});
          window.location.reload();
        }, 1000);
      }
      else
        document.getElementById("no-review-error").className = "popup-error collapse.show";
    }
    else
      document.getElementById("no-rating-error").className = "popup-error collapse.show";
  }

  closeNoRatingError() { document.getElementById("no-rating-error").className="popup-error collapse"; }

  closeNoReviewError() { document.getElementById("no-review-error").className="popup-error collapse"; }

  render() {
    return (
      <div id="boarding-form">
        <div className="row">
          <input className="search-bar" id="boarder-search" placeholder="Search..." maxLength="50" onChange={this.updateSearch} value={this.state.search}></input>
          <select className="search-dropdown" id="boarder-search-category" onChange={this.handleChange}>
            <option value="name">Name</option>
            <option value="address">Address</option>
            <option value="city">City</option>
            <option value="dailyRate">Daily Rate</option>
          </select>
        </div>
        <div className="pooch-title">Dog Boarders</div>
        <div id="loader" className="mb-4"><Loader type="ThreeDots" color="black" height={75} width={75}/></div>
        <div id="bubble-home" className="row"></div>

        <div id="boarder-popup" className="row collapse">
          
          <div className="col" id="popup-boarder-left">
              <div className="my-3 row">
                <div className="col"><img className="profile-pic mr-5" id="popup-boarder-pic" src="" alt="Profile Picture"/></div>
                <div className="col"><div id="popup-boarder-rating"/></div>
              </div>
              <div className="my-3 row"><div className="popup-input collapse" id="popup-boarder-email"/></div>
              <div className="my-3 row"><div className="popup-input" id="popup-boarder-name"/></div>
              <div className="my-3 row"><div className="popup-input" id="popup-boarder-phone"/></div>
              <div className="my-3 row"><div className="popup-input" id="popup-boarder-city"/></div>
              <div className="my-3 row"><div className="popup-input" id="popup-boarder-daily-rate"/></div>
              <div className="col" id="popup-boarder-images">
                <img className="popup-feature collapse" id="popup-boarder-feature1-pic" alt="Featured Picture"/>
                <img className="popup-feature collapse" id="popup-boarder-feature2-pic" alt="Featured Picture"/>
                <img className="popup-feature collapse" id="popup-boarder-feature3-pic" alt="Featured Picture"/>
                <img className="popup-feature collapse" id="popup-boarder-feature4-pic" alt="Featured Picture"/>
              </div>
          </div>
          <div className="col" id="popup-boarder-reviews">
              <img src={Close} className="pull-right" id="close-popup-button" onClick={this.closeProfile}/>
              <div className="popup-new-rating row justify-content-center mt-5 mb-2">
                <div className="empty-stars" id="star1" onClick={this.fillStar}>☆</div>
                <div className="empty-stars" id="star2" onClick={this.fillStar}>☆</div>
                <div className="empty-stars" id="star3" onClick={this.fillStar}>☆</div>
                <div className="empty-stars" id="star4" onClick={this.fillStar}>☆</div>
                <div className="empty-stars" id="star5" onClick={this.fillStar}>☆</div>
              </div>
              <textarea className="mt-2 mb-1" id="new-review" placeholder="Write a review..." maxLength="256"></textarea>
              <button id="submit-review" className="btn mb-3" onClick={this.submitReview}>Submit Review</button>
              <div id="review-col"></div>
          </div>
        </div>

        <div className="popup-error collapse" id="no-rating-error">
          Select a rating
          <div className="my-2"><button className="error-button btn-danger" onClick={this.closeNoRatingError}>Dismiss</button></div>
        </div>
        <div className="popup-error collapse" id="no-review-error">
          Review can't be empty
          <div className="my-2"><button className="error-button btn-danger" onClick={this.closeNoReviewError}>Dismiss</button></div>
        </div>
      </div>
    );
  }
}

export default BoardingForm;
