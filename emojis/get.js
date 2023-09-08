const run = async () => {
var page = 1
var next = true
while (next == true) {
    // get https://discadia.com/api/emojis?limit=100&page=page
    var response = await fetch(`https://discadia.com/api/emojis?limit=100&page=${page}`)
    var json = await response.json()
    var urls = json.data.map((emoji) => emoji.resized_url.replace("https://emoji.discadia.com/emojis/resized/", ""))
    // print urls to file
    console.log(urls.join('\n'))
    next = json.meta.has_next
    // next = false
    console.warn(`Page ${page} done`)
    page += 1
}
}


run()