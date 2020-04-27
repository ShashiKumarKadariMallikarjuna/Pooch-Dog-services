import React from "react";
import Fire from "../../config/Fire.js"; // Firebase
import "./styles.scss"; // Styles
import Loader from "react-loader-spinner"; // Animation
// Graphics
import Gray1 from "./graydog1.png";
import Gray2 from "./graydog2.png";
import Gray3 from "./graydog3.png";
import AddButton from "./add-button.png";

class PetForm extends React.Component {

  componentDidMount() {
    document.getElementById("pet-loader").style.display = "block";
    
    setTimeout(() => { // Check if user has dog profiles
      Fire.firestore().collection("pets")
      .where("email", "==", Fire.auth().currentUser.email)
      .get()
      .then(snapshot => {
        snapshot.docs.forEach(doc => {
          //Hide initial setup elements
          for (var i=0; i<document.getElementsByClassName("dog-icon gray").length; i++)
            document.getElementsByClassName("dog-icon gray")[i].className = "dog-icon gray collapse";
          document.getElementById("pet-welcome").className = "collapse";
          document.getElementById("pet-welcome-arrow").className = "collapse";
          document.getElementById("open-dog-profile-tip").className = "pooch-navbar-item collapse.show";
          // Apend dog icon row with next dog pictue
          var nextPic = document.createElement("img");
          nextPic.id = doc.data().name+"-picture";
          nextPic.className = "dog-icon";
          nextPic.alt = "dog-profile-picture-preview";
          nextPic.src = doc.data().pic;
          nextPic.addEventListener("click", ()=>{viewProfile(doc.data().name)});
          // document.getElementById("dog-icon-row").appendChild(nextPic);
          document.getElementById("dog-icon-row").prepend(nextPic);
        });
      });
      document.getElementById("pet-loader").style.display = "none";
    }, 1000);

    function viewProfile(dogName) {
      
        Fire.firestore().collection("pets")
        .where("email", "==", Fire.auth().currentUser.email).where("name", "==", dogName.toLowerCase())
        .get()
        .then(snapshot => {
          snapshot.docs.forEach(doc => {
            var simpleBreed = false;
            for (var i=0; i<document.getElementById("database-dog-breed-simple").length; i++) {
              if (document.getElementById("database-dog-breed-simple").options[i].value === doc.data().breed)
                simpleBreed = true;
            }
            if (simpleBreed === true) {
              document.getElementById("database-dog-breed-simple").value = doc.data().breed;
              document.getElementById("database-dog-breed-simple-row").className = "my-4 row collapse.show";
              document.getElementById("database-dog-breed-row").className = "my-4 row collapse";
              document.getElementById("database-breed-checkbox").checked = true;
            }
            else {
              document.getElementById("database-dog-breed").value = doc.data().breed;
              document.getElementById("database-dog-breed-simple-row").className = "my-4 row collapse";
              document.getElementById("database-dog-breed-row").className = "my-4 row collapse.show";
              document.getElementById("database-breed-checkbox").checked = false;
            }
            // Make name camelcase
            var nameWords = doc.data().name.toLowerCase().split(" ");
            var adjustedName = "";
            for (var i=0; i<nameWords.length; i++)
              adjustedName += nameWords[i].substr(0,1).toUpperCase() + nameWords[i].substr(1, nameWords[i].length-1) + " ";
            document.getElementById("database-dog-name").value = adjustedName;
            document.getElementById("database-dog-gender").value = doc.data().gender; 
            document.getElementById("database-dog-color").value = doc.data().color;
            document.getElementById("database-dog-age").value = doc.data().age;
            document.getElementById("database-dog-profile-pic").src = doc.data().pic;
            document.getElementById("database-profile").className = "collapse.show";
            document.getElementById("new-dog-profile").className = "collapse";
            
            document.getElementById("delete-dog-button-row").className = "collapse.show";
          });
        });
    }
  }

newDog() {
  
  Fire.firestore().collection("pets").where('email', '==', Fire.auth().currentUser.email)
  .where("name", "==", document.getElementById("dog-name").value)
  .get().then(function(querySnapshot) {
    querySnapshot.forEach(function(doc) {
      doc.ref.delete();
    });
  });
  setTimeout(() => {
  if (document.getElementById("breed-checkbox").checked === true) {
    console.log("adding from simple dog breeds");
    Fire.firestore().collection("pets").add({
      email: Fire.auth().currentUser.email,
      name: document.getElementById("dog-name").value.toLowerCase(),
      gender: document.getElementById("dog-gender").value,
      breed: document.getElementById("dog-breed-simple").value,
      color: document.getElementById("dog-color").value,
      age: document.getElementById("dog-age").value,
      pic: document.getElementById("dog-profile-pic").src
  });
  }
  else if (document.getElementById("breed-checkbox").checked === false) {
    console.log("adding from all dog breeds");
    Fire.firestore().collection("pets").add({
      email: Fire.auth().currentUser.email,
      name: document.getElementById("dog-name").value.toLowerCase(),
      gender: document.getElementById("dog-gender").value,
      breed: document.getElementById("dog-breed").value,
      color: document.getElementById("dog-color").value,
      age: document.getElementById("dog-age").value,
      pic: document.getElementById("dog-profile-pic").src
    });
  }
}, 1000);
  setTimeout(() => {
    window.location.reload();
  }, 2000);
}

updateDog() {
  Fire.firestore().collection("pets").where('email', '==', Fire.auth().currentUser.email)
  .where("name", "==", document.getElementById("database-dog-name").value)
  .get().then(function(querySnapshot) {
    querySnapshot.forEach(function(doc) {
      doc.ref.delete();
    });
  });
  setTimeout(() => {
  if (document.getElementById("breed-checkbox").checked === true) {
    console.log("adding from simple dog breeds");
    Fire.firestore().collection("pets").add({
      email: Fire.auth().currentUser.email,
      name: document.getElementById("database-dog-name").value,
      gender: document.getElementById("database-dog-gender").value,
      breed: document.getElementById("database-dog-breed-simple").value,
      color: document.getElementById("database-dog-color").value,
      age: document.getElementById("database-dog-age").value,
      pic: document.getElementById("database-dog-profile-pic").src
  });
  }
  else if (document.getElementById("breed-checkbox").checked === false) {
    console.log("adding from all dog breeds");
    Fire.firestore().collection("pets").add({
      email: Fire.auth().currentUser.email,
      name: document.getElementById("database-dog-name").value,
      gender: document.getElementById("database-dog-gender").value,
      breed: document.getElementById("database-dog-breed").value,
      color: document.getElementById("database-dog-color").value,
      age: document.getElementById("database-dog-age").value,
      pic: document.getElementById("database-dog-profile-pic").src
    });
   }
}, 1000);
  setTimeout(() => {
    window.location.reload();
  }, 2000);
}

swapBreedList() {
  if (document.getElementById("breed-checkbox").checked === true) {
    document.getElementById("dog-breed-row").className = "my-4 row collapse";
    document.getElementById("dog-breed-simple-row").className = "my-4 row collapse.show";
  }
  else if (document.getElementById("breed-checkbox").checked === false) {
    document.getElementById("dog-breed-row").className = "my-4 row collapse.show";
    document.getElementById("dog-breed-simple-row").className = "my-4 row collapse";
  }
  if (document.getElementById("database-breed-checkbox").checked === true) {
    document.getElementById("database-dog-breed-row").className = "my-4 row collapse";
    document.getElementById("database-dog-breed-simple-row").className = "my-4 row collapse.show";
  }
  else if (document.getElementById("database-breed-checkbox").checked === false) {
    document.getElementById("database-dog-breed-row").className = "my-4 row collapse.show";
    document.getElementById("database-dog-breed-simple-row").className = "my-4 row collapse";
  }
}

previewDogPic(event) {
  var input = event.target; // Get image that changed state of input element
  var reader = new FileReader();
  reader.onload = function() {
  var dataURL = reader.result;
  var output = document.getElementById('dog-profile-pic');
  output.src = dataURL;
  };
  reader.readAsDataURL(input.files[0]); // Show preview of image
}

previewDatabaseDogPic(event) {
  var input = event.target; // Get image that changed state of input element
  var reader = new FileReader();
  reader.onload = function() {
  var dataURL = reader.result;
  var output = document.getElementById('database-dog-profile-pic');
  output.src = dataURL;
  };
  reader.readAsDataURL(input.files[0]); // Show preview of image
}

toggleNewProfile() {
  if (document.getElementById("new-dog-profile").className === "collapse") {
    document.getElementById("database-profile").className = "collapse";
    document.getElementById("new-dog-profile").className = "collapse.show";
    document.getElementById("delete-dog-button-row").className = "collapse";
  }
  else if (document.getElementById("new-dog-profile").className === "collapse.show")
    document.getElementById("new-dog-profile").className = "collapse";
}

openDeletePopup() {document.getElementById("delete-dog-popup").className="fixed-top collapse.show";}

closeDeletePopup() {document.getElementById("delete-dog-popup").className="fixed-top collapse";}

deleteProfile() {
  Fire.firestore().collection("pets")
  .where('email', '==', Fire.auth().currentUser.email)
  .where('name', '==', document.getElementById("database-dog-name").value)
  .get().then(function(querySnapshot) {
      querySnapshot.forEach(function(doc) {
        doc.ref.delete();
      });
    });
    setTimeout(() => {
      window.location.reload();
    }, 1000);
}

  render() {
    return (
      <div id="pet-form">
        <div className="my-5 row" id="dog-icon-row">
          <img className="dog-icon gray" alt="dog-icon" src={Gray1}/>
          <img className="dog-icon gray" alt="dog-icon" src={Gray2}/>
          <img className="dog-icon gray" alt="dog-icon" src={Gray3}/>
          <div className="pooch-navbar-item collapse" id="open-dog-profile-tip"><b>←</b> View dog profile</div>
          <div className="mt-2" id="pet-loader"><Loader type="TailSpin" color="black" height={75} width={75}/></div>
        </div>
        {/* <div className="dog-icon" id="add-dog-button" onClick={this.toggleNewProfile}>+</div> */}
        <img className="row" id="add-dog-button" alt="add-dog-button" src={AddButton} onClick={this.toggleNewProfile}/>
        <div className="pooch-navbar-item row" id="pet-welcome-arrow" style={{height: "7vw"}}>Add your dogs <b>⤣</b></div>
        <div className="pooch-navbar-item" id="pet-welcome">Here you can manage all of your pets' profiles</div>
        <div className="pooch-navbar-item collapse" id="open-dog-profile-tip"><b>	↖</b> View dog profile</div>
        <div id="new-dog-profile" className="collapse">
              <div className="mb-3"><img className="dog-pic" id="dog-profile-pic" src={Gray1} alt="Profile"/></div>
              <input type="file" id="dog-input" onChange={this.previewDogPic}/>
              <div className="col">
              <div className="my-4 row">
                <input className="profile-input" id="dog-name" type="text" placeholder="Name" maxLength="50"/>
              </div>
              <div className="my-4 row">
                  <select className="profile-input" id="dog-gender">
                      <option value="">Gender</option>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                    </select>
                </div>
                <div className="my-4 row collapse.show" id="dog-breed-simple-row">
                  <select className="profile-input" id="dog-breed-simple" name="Dog Breed:">
                      <option value="">Dog Breed</option>
                      <option value="Mixed">Mixed</option>
                      <option value="Australian Shepard">Australian Shepard</option>
                      <option value="Beagle">Beagle</option>
                      <option value="Boxer">Boxer</option>
                      <option value="Bulldog">Bulldog</option>
                      <option value="Chihuahua">Chihuahua</option>
                      <option value="Collie">Collie</option>
                      <option value="Corgie">Corgie</option>
                      <option value="Daschund">Daschund</option>
                      <option value="Dalmation">Dalmation</option>
                      <option value="German Shepard">German Shepard</option>
                      <option value="Golden Retreiver">Golden Retreiver</option>
                      <option value="Great Dane">Great Dane</option>
                      <option value="Hound">Hound</option>
                      <option value="Labrador Retreiver">Labrador Retreiver</option>
                      <option value="Mastiff">Mastiff</option>
                      <option value="Pinscher">Pinscher</option>
                      <option value="Pointer">Pointer</option>
                      <option value="Pomeranian">Pomeranian</option>
                      <option value="Poodle">Poodle</option>
                      <option value="Pug">Pug</option>
                      <option value="Rottweiler">Rottweiler</option>
                      <option value="Setter">Setter</option>
                      <option value="Siberian Husky">Siberian Husky</option>
                      <option value="Schnauzer">Schnauzer</option>
                      <option value="Sheepdog">Sheepdog</option>
                      <option value="Shih Tzu">Shih Tzu</option>
                      <option value="Spaniel">Spaniel</option>
                      <option value="Terrier">Terrier</option>
                  </select>
                </div>
                <div className="my-4 row collapse" id="dog-breed-row">
                  <select className="profile-input" id="dog-breed">
                      <option value="">Dog Breed (all)</option>
                      <option value="Affenpinscher">Affenpinscher</option>
                      <option value="Afghan Hound">Afghan Hound</option>
                      <option value="Airedale Terrier">Airedale Terrier</option>
                      <option value="Akita">Akita</option>
                      <option value="Alaskan Malamute">Alaskan Malamute</option>
                      <option value="American English Coonhound">American English Coonhound</option>
                      <option value="American Eskimo Dog">American Eskimo Dog</option>
                      <option value="American Foxhound">American Foxhound</option>
                      <option value="American Hairless Terrier">American Hairless Terrier</option>
                      <option value="American Leopard Hound">American Leopard Hound</option>
                      <option value="American Staffordshire Terrier">American Staffordshire Terrier</option>
                      <option value="American Water Spaniel">American Water Spaniel</option>
                      <option value="Anatolian Shepard Dog">Anatolian Shepard Dog</option>
                      <option value="Appenzeller Sennenhund">Appenzeller Sennenhund</option>
                      <option value="Australian Cattle Dog">Australian Cattle Dog</option>
                      <option value="Australian Kelpie">Australian Kelpie</option>
                      <option value="Australian Shepard">Australian Shepard</option>
                      <option value="Australian Terrier">Australian Terrier</option>
                      <option value="Azawakh">Azawakh</option>
                      <option value="Barbet">Barbet</option>
                      <option value="Basenji">Basenji</option>
                      <option value="Basset Hound">Basset Hound</option>
                      <option value="Bavarian Mountain Scent Hound">Bavarian Mountain Scent Hound</option>
                      <option value="Beagle">Beagle</option>
                      <option value="Bearded Collie">Bearded Collie</option>
                      <option value="Beauceron">Beauceron</option>
                      <option value="Bedlington Terrier">Bedlington Terrier</option>
                      <option value="Belgian Laekenois">Belgian Laekenois</option>
                      <option value="Belgian Malinois">Belgian Malinois</option>
                      <option value="Belgian Sheepdog">Belgian Sheepdog</option>
                      <option value="Beligan Tervuren">Beligan Tervuren</option>
                      <option value="Bergamasco Sheepdog">Bergamasco Sheepdog</option>
                      <option value="Berger Picard">Berger Picard</option>
                      <option value="Bernese Mountain Dog">Bernese Mountain Dog</option>
                      <option value="Bichon Frise">Bichon Frise</option>
                      <option value="Biewer Terrier">Biewer Terrier</option>
                      <option value="Black and Tan Coonhound">Black and Tan Coonhound</option>
                      <option value="Black Russian Terrier">Black Russian Terrier</option>
                      <option value="Bloodhound">Bloodhound</option>
                      <option value="Bluetick Coonhound">Bluetick Coonhound</option>
                      <option value="Boerboel">Boerboel</option>
                      <option value="Bohemian Shepard">Bohemian Shepard</option>
                      <option value="Bolognese">Bolognese</option>
                      <option value="Border Collie">Border Collie</option>
                      <option value="Border Terrier">Border Terrier</option>
                      <option value="Borzoi">Borzoi</option>
                      <option value="Boston Terrier">Boston Terrier</option>
                      <option value="Boxer">Boxer</option>
                      <option value="Boykin Spaniel">Boykin Spaniel</option>
                      <option value="Bracco Italiano">Bracco Italiano</option>
                      <option value="Briard">Briard</option>
                      <option value="Brittany">Brittany</option>
                      <option value="Broholmer">Broholmer</option>
                      <option value="Brussels Griffon">Brussels Griffon</option>
                      <option value="Bull Terrier">Bull Terrier</option>
                      <option value="Bulldog">Bulldog</option>
                      <option value="Bullmastiff">Bullmastiff</option>
                      <option value="Cairn Terrier">Cairn Terrier</option>
                      <option value="Cannan Dog">Cannan Dog</option>
                      <option value="Cane Corso">Cane Corso</option>
                      <option value="Cardigan Welsh Corgi">Cardigan Welsh Corgi</option>
                      <option value="Carolina Dog">Carolina Dog</option>
                      <option value="Catahoula Leopard Dog">Catahoula Leopard Dog</option>
                      <option value="Caucasian Shepard Dog">Caucasian Shepard Dog</option>
                      <option value="Central Asian Shepard Dog">Central Asian Shepard Dog</option>
                      <option value="Cesky Terrier">Cesky Terrier</option>
                      <option value="Chesapeake Bay Retreiver">Chesapeake Bay Retreiver</option>
                      <option value="Chihuahua">Chihuahua</option>
                      <option value="Chinese Crested">Chinese Crested</option>
                      <option value="Chinese Sharpei">Chinese Sharpei</option>
                      <option value="Chinook">Chinook</option>
                      <option value="Chow Chow">Chow Chow</option>
                      <option value="Clumber Spaniel">Clumber Spaniel</option>
                      <option value="Cocker Spaniel">Cocker Spaniel</option>
                      <option value="Collie">Collie</option>
                      <option value="Croatian Sheepdog">Croatian Sheepdog</option>
                      <option value="Curly Coated Retreiver">Curly Coated Retreiver</option>
                      <option value="Daschund">Daschund</option>
                      <option value="Dalmation">Dalmation</option>
                      <option value="Dandie Dinmont Terrier">Dandie Dinmont Terrier</option>
                      <option value="Danish Swedish Farmdog">Danish Swedish Farmdog</option>
                      <option value="Deutcher Watchelhund">Deutcher Watchelhund</option>
                      <option value="Doberman Pinscher">Doberman Pinscher</option>
                      <option value="Dogo Argentino">Dogo Argentino</option>
                      <option value="Dogue de Bordeaux">Dogue de Bordeaux</option>
                      <option value="Drever">Drever</option>
                      <option value="Dutch Shepard">Dutch Shepard</option>
                      <option value="English Cocker Spaniel">English Cocker Spaniel</option>
                      <option value="English Foxhound">English Foxhound</option>
                      <option value="English Setter">English Setter</option>
                      <option value="English Springer Spaniel">English Springer Spaniel</option>
                      <option value="English Toy Spaniel">English Toy Spaniel</option>
                      <option value="Entlebucher Mountain Dog">Entlebucher Mountain Dog</option>
                      <option value="Estrela Mountain Dog">Estrela Mountain Dog</option>
                      <option value="Eurasier">Eurasier</option>
                      <option value="Field Spaniel">Field Spaniel</option>
                      <option value="Finnish Lapphund">Finnish Lapphund</option>
                      <option value="Finnish Spitz">Finnish Spitz</option>
                      <option value="Flat Coated Retreiver">Flat Coated Retreiver</option>
                      <option value="French Bulldog">French Bulldog</option>
                      <option value="French Spaniel">French Spaniel</option>
                      <option value="German Longhaired Pointer">German Longhaired Pointer</option>
                      <option value="German Pinscher">German Pinscher</option>
                      <option value="German Shepard">German Shepard</option>
                      <option value="German Shorthaired Pointer">German Shorthaired Pointer</option>
                      <option value="German Spitz">German Spitz</option>
                      <option value="German Wirehaired Pointer">German Wirehaired Pointer</option>
                      <option value="Giant Schnauzer">Giant Schnauzer</option>
                      <option value="Golden Retreiver">Golden Retreiver</option>
                      <option value="Gordon Setter">Gordon Setter</option>
                      <option value="Great Dane">Great Dane</option>
                      <option value="Great Pyrenees">Great Pyrenees</option>
                      <option value="Greater Swiss Mountain Dog">Greater Swiss Mountain Dog</option>
                      <option value="Greyhound">Greyhound</option>
                      <option value="Hamltonstovare">Hamltonstovare</option>
                      <option value="Hanoverian Scenthound">Hanoverian Scenthound</option>
                      <option value="Harrier">Harrier</option>
                      <option value="Havanese">Havanese</option>
                      <option value="Hokkaido">Hokkaido</option>
                      <option value="Hovawart">Hovawart</option>
                      <option value="Ibizian Hound">Ibizian Hound</option>
                      <option value="Icelandic Sheepdog">Icelandic Sheepdog</option>
                      <option value="Irish Red and White Setter">Irish Red and White Setter</option>
                      <option value="Irish Setter">Irish Setter</option>
                      <option value="Irish Terrier">Irish Terrier</option>
                      <option value="Irish Water Spaniel">Irish Water Spaniel</option>
                      <option value="Irish Wolfhound">Irish Wolfhound</option>
                      <option value="Italian Greyhound">Italian Greyhound</option>
                      <option value="Jagdterrier">Jagdterrier</option>
                      <option value="Japanese Chin">Japanese Chin</option>
                      <option value="Japanese Spitz">Japanese Spitz</option>
                      <option value="Jindo">Jindo</option>
                      <option value="Kai Ken">Kai Ken</option>
                      <option value="Karelian Bear Dog">Karelian Bear Dog</option>
                      <option value="Keeshond">Keeshond</option>
                      <option value="Kerry Blue Terrier">Kerry Blue Terrier</option>
                      <option value="Kishu Ken">Kishu Ken</option>
                      <option value="Komondor">Komondor</option>
                      <option value="Kuvasz">Kuvasz</option>
                      <option value="Labrador Retreiver">Labrador Retreiver</option>
                      <option value="Lakeland Terrier">Lakeland Terrier</option>
                      <option value="Lanchashire Heeler">Lanchashire Heeler</option>
                      <option value="Laponian Herder">Laponian Herder</option>
                      <option value="Leonberger">Leonberger</option>
                      <option value="Lhasa Apso">Lhasa Apso</option>
                      <option value="Lowchen">Lowchen</option>
                      <option value="Maltese">Maltese</option>
                      <option value="Manchester Terrier">Manchester Terrier</option>
                      <option value="Mastiff">Mastiff</option>
                      <option value="Miniature American Shepard">Miniature American Shepard</option>
                      <option value="Miniature Bull Terrier">Miniature Bull Terrier</option>
                      <option value="Miniature Pinscher">Miniature Pinscher</option>
                      <option value="Miniature Schnauzer">Miniature Schnauzer</option>
                      <option value="Mountain Cur">Mountain Cur</option>
                      <option value="Mudi">Mudi</option>
                      <option value="Neopolitan Mastiff">Neopolitan Mastiff</option>
                      <option value="Newfoundland">Newfoundland</option>
                      <option value="Norfolk Terrier">Norfolk Terrier</option>
                      <option value="Norweigan Buhund">Norweigan Buhund</option>
                      <option value="Norweigan Elkhound">Norweigan Elkhound</option>
                      <option value="Norweigan Lundehund">Norweigan Lundehund</option>
                      <option value="Norwich Terrier">Norwich Terrier</option>
                      <option value="Old English Sheepdog">Old English Sheepdog</option>
                      <option value="Otterhound">Otterhound</option>
                      <option value="Papillon">Papillon</option>
                      <option value="Parson Russel Terrier">Parson Russel Terrier</option>
                      <option value="Pekingese">Pekingese</option>
                      <option value="Pembroke Welsh Corgi">Pembroke Welsh Corgi</option>
                      <option value="Peruvian Inca Orchid">Peruvian Inca Orchid</option>
                      <option value="Pharoah Hound">Pharoah Hound</option>
                      <option value="Plott Hound">Plott Hound</option>
                      <option value="Pointer">Pointer</option>
                      <option value="Polish Lowland Sheepdog">Polish Lowland Sheepdog</option>
                      <option value="Pomeranian">Pomeranian</option>
                      <option value="Poodle">Poodle</option>
                      <option value="Porcelaine">Porcelaine</option>
                      <option value="Portugese Podengo">Portugese Podengo</option>
                      <option value="Portugese Pointer">Portugese Pointer</option>
                      <option value="Portugese Sheep Dog">Portugese Sheep Dog</option>
                      <option value="Portugese Water Dog">Portugese Water Dog</option>
                      <option value="Pudelpointer">Pudelpointer</option>
                      <option value="Pug">Pug</option>
                      <option value="Puli">Puli</option>
                      <option value="Pumi">Pumi</option>
                      <option value="Pyrenean Mastiff">Pyrenean Mastiff</option>
                      <option value="Pyrenean Shepard">Pyrenean Shepard</option>
                      <option value="Rat Terrier">Rat Terrier</option>
                      <option value="Redbone Coonhound">Redbone Coonhound</option>
                      <option value="Rhodesian Ridgeback">Rhodesian Ridgeback</option>
                      <option value="Rottweiler">Rottweiler</option>
                      <option value="Russel Terrier">Russel Terrier</option>
                      <option value="Russian Toy">Russian Toy</option>
                      <option value="Saint Bernard">Saint Bernard</option>
                      <option value="Saluki">Saluki</option>
                      <option value="Samoyed">Samoyed</option>
                      <option value="Schapendoes">Schapendoes</option>
                      <option value="Shipperke">Shipperke</option>
                      <option value="Scottish Deerhound">Scottish Deerhound</option>
                      <option value="Scottish Terrier">Scottish Terrier</option>
                      <option value="Sealyham Terrier">Sealyham Terrier</option>
                      <option value="Segugio Italiano">Segugio Italiano</option>
                      <option value="Shetland Sheepdog">Shetland Sheepdog</option>
                      <option value="Shiba Inu">Shiba Inu</option>
                      <option value="Shih Tzu">Shih Tzu</option>
                      <option value="Shikoku">Shikoku</option>
                      <option value="Siberian Husky">Siberian Husky</option>
                      <option value="Silky Terrier">Silky Terrier</option>
                      <option value="Skye Terrier">Skye Terrier</option>
                      <option value="Sloughi">Sloughi</option>
                      <option value="Slovakian Wirehaired Pointer">Slovakian Wirehaired Pointer</option>
                      <option value="Small Munsterlander Pointer">Small Munsterlander Pointer</option>
                      <option value="Smooth Fox Terrier">Smooth Fox Terrier</option>
                      <option value="Soft Coated Wheaton Terrier">Soft Coated Wheaton Terrier</option>
                      <option value="Spanish Mastiff">Spanish Mastiff</option>
                      <option value="Spanish Water Dog">Spanish Water Dog</option>
                      <option value="Spinone Italiano">Spinone Italiano</option>
                      <option value="Stabyhoun">Stabyhoun</option>
                      <option value="Staffordshire Bull Terrier">Staffordshire Bull Terrier</option>
                      <option value="Standard Schnauzer">Standard Schnauzer</option>
                      <option value="Sussex Spaniel">Sussex Spaniel</option>
                      <option value="Swedish Lapphund">Swedish Lapphund</option>
                      <option value="Swedish Vallhund">Swedish Vallhund</option>
                      <option value="Taiwan Dog">Taiwan Dog</option>
                      <option value="Teddy Roosevelt Terrier">Teddy Roosevelt Terrier</option>
                      <option value="Thai Ridgeback">Thai Ridgeback</option>
                      <option value="Tibetan Mastiff">Tibetan Mastiff</option>
                      <option value="Tibetan Spaniel">Tibetan Spaniel</option>
                      <option value="Tibetan Terrier">Tibetan Terrier</option>
                      <option value="Tornjak">Tornjak</option>
                      <option value="Tosa">Tosa</option>
                      <option value="Toy Fox Terrier">Toy Fox Terrier</option>
                      <option value="Translyvanian Hound">Translyvanian Hound</option>
                      <option value="Treeing Tennessee Brindle">Treeing Tennessee Brindle</option>
                      <option value="Treeing Walker Coonhound">Treeing Walker Coonhound</option>
                      <option value="Vizsla">Vizsla</option>
                      <option value="Weimaraner">Weimaraner</option>
                      <option value="Welsh Springer Spaniel">Welsh Springer Spaniel</option>
                      <option value="Welsh Terrier">Welsh Terrier</option>
                      <option value="Whippet">Whippet</option>
                      <option value="Wire Fox Terrier">Wire Fox Terrier</option>
                      <option value="Wirehaired Pointing Griffon">Wirehaired Pointing Griffon</option>
                      <option value="Wirehaired Vizsla">Wirehaired Vizsla</option>
                      <option value="Working Kelpie">Working Kelpie</option>
                      <option value="Xoloitzcuintli">Xoloitzcuintli</option>
                      <option value="Yakutian Laika">Yakutian Laika</option>
                      <option value="Yorkshire Terrier">Yorkshire Terrier</option>
                  </select>
                </div>
                <div className="my-4 row">
                  <select className="profile-input" id="dog-color">
                      <option value="">Select color</option>
                      <option value="Black">Black</option>
                      <option value="Grey">Grey</option>
                      <option value="Blue">Blue</option>
                      <option value="White">White</option>
                      <option value="Yellow">Yellow</option>
                      <option value="Orange">Orange</option>
                      <option value="Red">Red</option>
                      <option value="Brown">Brown</option>
                   </select>
                </div> 
                <div className="my-4 row">
                  <input className="profile-input" id="dog-age" type="number" placeholder="Age" min="0" max="30"/>
                </div>
                <div className="my-4 row trak_body-small">
                  <input type="checkbox" id="breed-checkbox" className="mr-4 my-2" onChange={this.swapBreedList} defaultChecked></input>Show only common dog breeds
                </div>
              </div>
              <button type="submit" className="btn btn-primary" onClick={this.newDog}>Create Dog Profile</button>
          </div>

          <div id="database-profile" className="collapse">
              <div className="mb-3"><img className="dog-pic" id="database-dog-profile-pic" alt="Profile"/></div>
              <input type="file" id="database-dog-input" onChange={this.previewDatabaseDogPic}/>
              <div className="col">
              <div className="my-4 row">
                <input className="profile-input" id="database-dog-name" type="text" placeholder="Name" maxLength="50"/>
              </div>
              <div className="my-4 row">
                  <select className="profile-input" id="database-dog-gender">
                      <option value="">Gender</option>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                    </select>
                </div>
                <div className="my-4 row collapse.show" id="database-dog-breed-simple-row">
                  <select className="profile-input" id="database-dog-breed-simple" name="Dog Breed:">
                      <option value="">Dog Breed</option>
                      <option value="Mixed">Mixed</option>
                      <option value="Australian Shepard">Australian Shepard</option>
                      <option value="Beagle">Beagle</option>
                      <option value="Boxer">Boxer</option>
                      <option value="Bulldog">Bulldog</option>
                      <option value="Chihuahua">Chihuahua</option>
                      <option value="Collie">Collie</option>
                      <option value="Corgie">Corgie</option>
                      <option value="Daschund">Daschund</option>
                      <option value="Dalmation">Dalmation</option>
                      <option value="German Shepard">German Shepard</option>
                      <option value="Golden Retreiver">Golden Retreiver</option>
                      <option value="Great Dane">Great Dane</option>
                      <option value="Hound">Hound</option>
                      <option value="Labrador Retreiver">Labrador Retreiver</option>
                      <option value="Mastiff">Mastiff</option>
                      <option value="Pinscher">Pinscher</option>
                      <option value="Pointer">Pointer</option>
                      <option value="Pomeranian">Pomeranian</option>
                      <option value="Poodle">Poodle</option>
                      <option value="Pug">Pug</option>
                      <option value="Rottweiler">Rottweiler</option>
                      <option value="Setter">Setter</option>
                      <option value="Siberian Husky">Siberian Husky</option>
                      <option value="Schnauzer">Schnauzer</option>
                      <option value="Sheepdog">Sheepdog</option>
                      <option value="Shih Tzu">Shih Tzu</option>
                      <option value="Spaniel">Spaniel</option>
                      <option value="Terrier">Terrier</option>
                  </select>
                </div>
                <div className="my-4 row collapse" id="database-dog-breed-row">
                  <select className="profile-input" id="database-dog-breed">
                      <option value="">Dog Breed (all)</option>
                      <option value="Affenpinscher">Affenpinscher</option>
                      <option value="Afghan Hound">Afghan Hound</option>
                      <option value="Airedale Terrier">Airedale Terrier</option>
                      <option value="Akita">Akita</option>
                      <option value="Alaskan Malamute">Alaskan Malamute</option>
                      <option value="American English Coonhound">American English Coonhound</option>
                      <option value="American Eskimo Dog">American Eskimo Dog</option>
                      <option value="American Foxhound">American Foxhound</option>
                      <option value="American Hairless Terrier">American Hairless Terrier</option>
                      <option value="American Leopard Hound">American Leopard Hound</option>
                      <option value="American Staffordshire Terrier">American Staffordshire Terrier</option>
                      <option value="American Water Spaniel">American Water Spaniel</option>
                      <option value="Anatolian Shepard Dog">Anatolian Shepard Dog</option>
                      <option value="Appenzeller Sennenhund">Appenzeller Sennenhund</option>
                      <option value="Australian Cattle Dog">Australian Cattle Dog</option>
                      <option value="Australian Kelpie">Australian Kelpie</option>
                      <option value="Australian Shepard">Australian Shepard</option>
                      <option value="Australian Terrier">Australian Terrier</option>
                      <option value="Azawakh">Azawakh</option>
                      <option value="Barbet">Barbet</option>
                      <option value="Basenji">Basenji</option>
                      <option value="Basset Hound">Basset Hound</option>
                      <option value="Bavarian Mountain Scent Hound">Bavarian Mountain Scent Hound</option>
                      <option value="Beagle">Beagle</option>
                      <option value="Bearded Collie">Bearded Collie</option>
                      <option value="Beauceron">Beauceron</option>
                      <option value="Bedlington Terrier">Bedlington Terrier</option>
                      <option value="Belgian Laekenois">Belgian Laekenois</option>
                      <option value="Belgian Malinois">Belgian Malinois</option>
                      <option value="Belgian Sheepdog">Belgian Sheepdog</option>
                      <option value="Beligan Tervuren">Beligan Tervuren</option>
                      <option value="Bergamasco Sheepdog">Bergamasco Sheepdog</option>
                      <option value="Berger Picard">Berger Picard</option>
                      <option value="Bernese Mountain Dog">Bernese Mountain Dog</option>
                      <option value="Bichon Frise">Bichon Frise</option>
                      <option value="Biewer Terrier">Biewer Terrier</option>
                      <option value="Black and Tan Coonhound">Black and Tan Coonhound</option>
                      <option value="Black Russian Terrier">Black Russian Terrier</option>
                      <option value="Bloodhound">Bloodhound</option>
                      <option value="Bluetick Coonhound">Bluetick Coonhound</option>
                      <option value="Boerboel">Boerboel</option>
                      <option value="Bohemian Shepard">Bohemian Shepard</option>
                      <option value="Bolognese">Bolognese</option>
                      <option value="Border Collie">Border Collie</option>
                      <option value="Border Terrier">Border Terrier</option>
                      <option value="Borzoi">Borzoi</option>
                      <option value="Boston Terrier">Boston Terrier</option>
                      <option value="Boxer">Boxer</option>
                      <option value="Boykin Spaniel">Boykin Spaniel</option>
                      <option value="Bracco Italiano">Bracco Italiano</option>
                      <option value="Briard">Briard</option>
                      <option value="Brittany">Brittany</option>
                      <option value="Broholmer">Broholmer</option>
                      <option value="Brussels Griffon">Brussels Griffon</option>
                      <option value="Bull Terrier">Bull Terrier</option>
                      <option value="Bulldog">Bulldog</option>
                      <option value="Bullmastiff">Bullmastiff</option>
                      <option value="Cairn Terrier">Cairn Terrier</option>
                      <option value="Cannan Dog">Cannan Dog</option>
                      <option value="Cane Corso">Cane Corso</option>
                      <option value="Cardigan Welsh Corgi">Cardigan Welsh Corgi</option>
                      <option value="Carolina Dog">Carolina Dog</option>
                      <option value="Catahoula Leopard Dog">Catahoula Leopard Dog</option>
                      <option value="Caucasian Shepard Dog">Caucasian Shepard Dog</option>
                      <option value="Central Asian Shepard Dog">Central Asian Shepard Dog</option>
                      <option value="Cesky Terrier">Cesky Terrier</option>
                      <option value="Chesapeake Bay Retreiver">Chesapeake Bay Retreiver</option>
                      <option value="Chihuahua">Chihuahua</option>
                      <option value="Chinese Crested">Chinese Crested</option>
                      <option value="Chinese Sharpei">Chinese Sharpei</option>
                      <option value="Chinook">Chinook</option>
                      <option value="Chow Chow">Chow Chow</option>
                      <option value="Clumber Spaniel">Clumber Spaniel</option>
                      <option value="Cocker Spaniel">Cocker Spaniel</option>
                      <option value="Collie">Collie</option>
                      <option value="Croatian Sheepdog">Croatian Sheepdog</option>
                      <option value="Curly Coated Retreiver">Curly Coated Retreiver</option>
                      <option value="Daschund">Daschund</option>
                      <option value="Dalmation">Dalmation</option>
                      <option value="Dandie Dinmont Terrier">Dandie Dinmont Terrier</option>
                      <option value="Danish Swedish Farmdog">Danish Swedish Farmdog</option>
                      <option value="Deutcher Watchelhund">Deutcher Watchelhund</option>
                      <option value="Doberman Pinscher">Doberman Pinscher</option>
                      <option value="Dogo Argentino">Dogo Argentino</option>
                      <option value="Dogue de Bordeaux">Dogue de Bordeaux</option>
                      <option value="Drever">Drever</option>
                      <option value="Dutch Shepard">Dutch Shepard</option>
                      <option value="English Cocker Spaniel">English Cocker Spaniel</option>
                      <option value="English Foxhound">English Foxhound</option>
                      <option value="English Setter">English Setter</option>
                      <option value="English Springer Spaniel">English Springer Spaniel</option>
                      <option value="English Toy Spaniel">English Toy Spaniel</option>
                      <option value="Entlebucher Mountain Dog">Entlebucher Mountain Dog</option>
                      <option value="Estrela Mountain Dog">Estrela Mountain Dog</option>
                      <option value="Eurasier">Eurasier</option>
                      <option value="Field Spaniel">Field Spaniel</option>
                      <option value="Finnish Lapphund">Finnish Lapphund</option>
                      <option value="Finnish Spitz">Finnish Spitz</option>
                      <option value="Flat Coated Retreiver">Flat Coated Retreiver</option>
                      <option value="French Bulldog">French Bulldog</option>
                      <option value="French Spaniel">French Spaniel</option>
                      <option value="German Longhaired Pointer">German Longhaired Pointer</option>
                      <option value="German Pinscher">German Pinscher</option>
                      <option value="German Shepard">German Shepard</option>
                      <option value="German Shorthaired Pointer">German Shorthaired Pointer</option>
                      <option value="German Spitz">German Spitz</option>
                      <option value="German Wirehaired Pointer">German Wirehaired Pointer</option>
                      <option value="Giant Schnauzer">Giant Schnauzer</option>
                      <option value="Golden Retreiver">Golden Retreiver</option>
                      <option value="Gordon Setter">Gordon Setter</option>
                      <option value="Great Dane">Great Dane</option>
                      <option value="Great Pyrenees">Great Pyrenees</option>
                      <option value="Greater Swiss Mountain Dog">Greater Swiss Mountain Dog</option>
                      <option value="Greyhound">Greyhound</option>
                      <option value="Hamltonstovare">Hamltonstovare</option>
                      <option value="Hanoverian Scenthound">Hanoverian Scenthound</option>
                      <option value="Harrier">Harrier</option>
                      <option value="Havanese">Havanese</option>
                      <option value="Hokkaido">Hokkaido</option>
                      <option value="Hovawart">Hovawart</option>
                      <option value="Ibizian Hound">Ibizian Hound</option>
                      <option value="Icelandic Sheepdog">Icelandic Sheepdog</option>
                      <option value="Irish Red and White Setter">Irish Red and White Setter</option>
                      <option value="Irish Setter">Irish Setter</option>
                      <option value="Irish Terrier">Irish Terrier</option>
                      <option value="Irish Water Spaniel">Irish Water Spaniel</option>
                      <option value="Irish Wolfhound">Irish Wolfhound</option>
                      <option value="Italian Greyhound">Italian Greyhound</option>
                      <option value="Jagdterrier">Jagdterrier</option>
                      <option value="Japanese Chin">Japanese Chin</option>
                      <option value="Japanese Spitz">Japanese Spitz</option>
                      <option value="Jindo">Jindo</option>
                      <option value="Kai Ken">Kai Ken</option>
                      <option value="Karelian Bear Dog">Karelian Bear Dog</option>
                      <option value="Keeshond">Keeshond</option>
                      <option value="Kerry Blue Terrier">Kerry Blue Terrier</option>
                      <option value="Kishu Ken">Kishu Ken</option>
                      <option value="Komondor">Komondor</option>
                      <option value="Kuvasz">Kuvasz</option>
                      <option value="Labrador Retreiver">Labrador Retreiver</option>
                      <option value="Lakeland Terrier">Lakeland Terrier</option>
                      <option value="Lanchashire Heeler">Lanchashire Heeler</option>
                      <option value="Laponian Herder">Laponian Herder</option>
                      <option value="Leonberger">Leonberger</option>
                      <option value="Lhasa Apso">Lhasa Apso</option>
                      <option value="Lowchen">Lowchen</option>
                      <option value="Maltese">Maltese</option>
                      <option value="Manchester Terrier">Manchester Terrier</option>
                      <option value="Mastiff">Mastiff</option>
                      <option value="Miniature American Shepard">Miniature American Shepard</option>
                      <option value="Miniature Bull Terrier">Miniature Bull Terrier</option>
                      <option value="Miniature Pinscher">Miniature Pinscher</option>
                      <option value="Miniature Schnauzer">Miniature Schnauzer</option>
                      <option value="Mountain Cur">Mountain Cur</option>
                      <option value="Mudi">Mudi</option>
                      <option value="Neopolitan Mastiff">Neopolitan Mastiff</option>
                      <option value="Newfoundland">Newfoundland</option>
                      <option value="Norfolk Terrier">Norfolk Terrier</option>
                      <option value="Norweigan Buhund">Norweigan Buhund</option>
                      <option value="Norweigan Elkhound">Norweigan Elkhound</option>
                      <option value="Norweigan Lundehund">Norweigan Lundehund</option>
                      <option value="Norwich Terrier">Norwich Terrier</option>
                      <option value="Old English Sheepdog">Old English Sheepdog</option>
                      <option value="Otterhound">Otterhound</option>
                      <option value="Papillon">Papillon</option>
                      <option value="Parson Russel Terrier">Parson Russel Terrier</option>
                      <option value="Pekingese">Pekingese</option>
                      <option value="Pembroke Welsh Corgi">Pembroke Welsh Corgi</option>
                      <option value="Peruvian Inca Orchid">Peruvian Inca Orchid</option>
                      <option value="Pharoah Hound">Pharoah Hound</option>
                      <option value="Plott Hound">Plott Hound</option>
                      <option value="Pointer">Pointer</option>
                      <option value="Polish Lowland Sheepdog">Polish Lowland Sheepdog</option>
                      <option value="Pomeranian">Pomeranian</option>
                      <option value="Poodle">Poodle</option>
                      <option value="Porcelaine">Porcelaine</option>
                      <option value="Portugese Podengo">Portugese Podengo</option>
                      <option value="Portugese Pointer">Portugese Pointer</option>
                      <option value="Portugese Sheep Dog">Portugese Sheep Dog</option>
                      <option value="Portugese Water Dog">Portugese Water Dog</option>
                      <option value="Pudelpointer">Pudelpointer</option>
                      <option value="Pug">Pug</option>
                      <option value="Puli">Puli</option>
                      <option value="Pumi">Pumi</option>
                      <option value="Pyrenean Mastiff">Pyrenean Mastiff</option>
                      <option value="Pyrenean Shepard">Pyrenean Shepard</option>
                      <option value="Rat Terrier">Rat Terrier</option>
                      <option value="Redbone Coonhound">Redbone Coonhound</option>
                      <option value="Rhodesian Ridgeback">Rhodesian Ridgeback</option>
                      <option value="Rottweiler">Rottweiler</option>
                      <option value="Russel Terrier">Russel Terrier</option>
                      <option value="Russian Toy">Russian Toy</option>
                      <option value="Saint Bernard">Saint Bernard</option>
                      <option value="Saluki">Saluki</option>
                      <option value="Samoyed">Samoyed</option>
                      <option value="Schapendoes">Schapendoes</option>
                      <option value="Shipperke">Shipperke</option>
                      <option value="Scottish Deerhound">Scottish Deerhound</option>
                      <option value="Scottish Terrier">Scottish Terrier</option>
                      <option value="Sealyham Terrier">Sealyham Terrier</option>
                      <option value="Segugio Italiano">Segugio Italiano</option>
                      <option value="Shetland Sheepdog">Shetland Sheepdog</option>
                      <option value="Shiba Inu">Shiba Inu</option>
                      <option value="Shih Tzu">Shih Tzu</option>
                      <option value="Shikoku">Shikoku</option>
                      <option value="Siberian Husky">Siberian Husky</option>
                      <option value="Silky Terrier">Silky Terrier</option>
                      <option value="Skye Terrier">Skye Terrier</option>
                      <option value="Sloughi">Sloughi</option>
                      <option value="Slovakian Wirehaired Pointer">Slovakian Wirehaired Pointer</option>
                      <option value="Small Munsterlander Pointer">Small Munsterlander Pointer</option>
                      <option value="Smooth Fox Terrier">Smooth Fox Terrier</option>
                      <option value="Soft Coated Wheaton Terrier">Soft Coated Wheaton Terrier</option>
                      <option value="Spanish Mastiff">Spanish Mastiff</option>
                      <option value="Spanish Water Dog">Spanish Water Dog</option>
                      <option value="Spinone Italiano">Spinone Italiano</option>
                      <option value="Stabyhoun">Stabyhoun</option>
                      <option value="Staffordshire Bull Terrier">Staffordshire Bull Terrier</option>
                      <option value="Standard Schnauzer">Standard Schnauzer</option>
                      <option value="Sussex Spaniel">Sussex Spaniel</option>
                      <option value="Swedish Lapphund">Swedish Lapphund</option>
                      <option value="Swedish Vallhund">Swedish Vallhund</option>
                      <option value="Taiwan Dog">Taiwan Dog</option>
                      <option value="Teddy Roosevelt Terrier">Teddy Roosevelt Terrier</option>
                      <option value="Thai Ridgeback">Thai Ridgeback</option>
                      <option value="Tibetan Mastiff">Tibetan Mastiff</option>
                      <option value="Tibetan Spaniel">Tibetan Spaniel</option>
                      <option value="Tibetan Terrier">Tibetan Terrier</option>
                      <option value="Tornjak">Tornjak</option>
                      <option value="Tosa">Tosa</option>
                      <option value="Toy Fox Terrier">Toy Fox Terrier</option>
                      <option value="Translyvanian Hound">Translyvanian Hound</option>
                      <option value="Treeing Tennessee Brindle">Treeing Tennessee Brindle</option>
                      <option value="Treeing Walker Coonhound">Treeing Walker Coonhound</option>
                      <option value="Vizsla">Vizsla</option>
                      <option value="Weimaraner">Weimaraner</option>
                      <option value="Welsh Springer Spaniel">Welsh Springer Spaniel</option>
                      <option value="Welsh Terrier">Welsh Terrier</option>
                      <option value="Whippet">Whippet</option>
                      <option value="Wire Fox Terrier">Wire Fox Terrier</option>
                      <option value="Wirehaired Pointing Griffon">Wirehaired Pointing Griffon</option>
                      <option value="Wirehaired Vizsla">Wirehaired Vizsla</option>
                      <option value="Working Kelpie">Working Kelpie</option>
                      <option value="Xoloitzcuintli">Xoloitzcuintli</option>
                      <option value="Yakutian Laika">Yakutian Laika</option>
                      <option value="Yorkshire Terrier">Yorkshire Terrier</option>
                  </select>
                </div>
                <div className="my-4 row">
                  <select className="profile-input" id="database-dog-color">
                      <option value="">Select color</option>
                      <option value="Black">Black</option>
                      <option value="Grey">Grey</option>
                      <option value="Blue">Blue</option>
                      <option value="White">White</option>
                      <option value="Yellow">Yellow</option>
                      <option value="Orange">Orange</option>
                      <option value="Red">Red</option>
                      <option value="Brown">Brown</option>
                   </select>
                </div> 
                <div className="my-4 row">
                  <input className="profile-input" id="database-dog-age" type="number" placeholder="Age" min="0" max="30"/>
                </div>
                <div className="my-4 row trak_body-small">
                  <input type="checkbox" id="database-breed-checkbox" className="mr-4 my-2" onChange={this.swapBreedList} defaultChecked></input>Show only common dog breeds
                </div>
              </div>
              <button type="submit" className="btn btn-primary" onClick={this.updateDog}>Update Dog Profile</button>
          </div>

          <div id="delete-dog-popup" className="fixed-top collapse">
              <div className="pooch-navbar-item">Are you sure you want to delete the current dog profile?</div>
              <div><button className="my-2 mr-4 btn-danger popup-button" onClick={this.deleteProfile}>Yes</button><button className="my-2 ml-4 btn-primary popup-button" onClick={this.closeDeletePopup}>No</button></div>
              <div className="trak_body">WARNING: This is irreversable!</div>
          </div>
          <div id="delete-dog-button-row" className="collapse"><button type="submit" className="btn btn-danger mt-8" onClick={this.openDeletePopup}>Delete Dog Profile</button></div>
      </div>
    );
  }
}

export default PetForm;
