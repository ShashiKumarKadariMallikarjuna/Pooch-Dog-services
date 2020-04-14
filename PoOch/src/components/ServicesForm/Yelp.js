const apiKey="jmi-PoyyCL6TczqHp0M9e5ywy7Bs8GAaEehHDP-7ktNpoflo4uvMUs-t312lgSqo8Ton8MVJ5faipw85aJGCk1O1YYZkWAPUMBy8Q8KkjvhabSflVGKZS65cDNJRXnYx";

const Yelp = {
    async search(term, location, radius ) {
        const e_rep = await fetch(`https://cors-anywhere.herokuapp.com/https://api.yelp.com/v3/events?location=${location}`, { headers: { Authorization: `Bearer ${apiKey}` } });
        console.log(e_rep.json());

        const response = await fetch(`https://cors-anywhere.herokuapp.com/https://api.yelp.com/v3/businesses/search?term=${term}&location=${location}&range=${radius}&limit=4`, { headers: { Authorization: `Bearer ${apiKey}` } });
          const jsonResponse = await response.json();
          console.log(jsonResponse)
          if (jsonResponse.businesses) {
              return jsonResponse.businesses.map(business => {
                  return {
                      id: business.id,
                      imageSrc: business.image_url,
                      name: business.name,
                      address: business.location.address1,
                      city: business.location.city,
                      state: business.location.state,
                      zipCode: business.location.zip_code,
                      category: business.categories[0].title,
                      rating: business.rating,
                      reviewCount: business.review_count,
                      url: business.url
                  };
              });
          }
     }
}