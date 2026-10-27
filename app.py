                        type="hidden",
                        name="file_hash",
                        value=file_hash,
                    ),
                    Label(
                        "Type this exact filename to confirm: "
                        + expected_name
                    ),
                    Input(
                        type="text",
                        name="confirm_name",
                        placeholder="Type the exact filename",
                        required=True,
                    ),
                    Button(
                        "Confirm permanent deletion",
                        type="submit",
                    ),
                    method="post",
                    action="/delete",
                ),
                P(A("Cancel", href="/manage")),
                cls="panel",
            )
        ),
    )


# ==================================================
# DELETE SELECTED DOCUMENT CHUNKS
# ==================================================

@rt("/delete", methods=["POST"])
def delete_document(file_hash: str, confirm_name: str):
    chunks = get_chunks_for_file(file_hash)

    if not chunks:
        return Titled(
            "Nothing deleted",
            P("No matching chunks were found."),
            A("Return to management", href="/manage"),
        )

    names = sorted({
        str((doc.meta or {}).get("source"))
        for doc in chunks
        if (doc.meta or {}).get("source")
    })

    expected_name = ", ".join(names) or "Unknown filename"

    if confirm_name.strip() != expected_name:
        return Titled(
            "Confirmation did not match",
            P("The filename did not match. Nothing was deleted."),
            P(f"Expected filename: {expected_name}"),
            A("Return to management", href="/manage"),
        )

    # Verify every chunk before deleting anything.
    if any(
        (doc.meta or {}).get("file_hash") != file_hash
        for doc in chunks
    ):
        return Titled(
            "Safety check failed",
            P("Nothing was deleted because validation failed."),
            A("Return to management", href="/manage"),
        )

    ids_to_delete = [doc.id for doc in chunks]

    if not ids_to_delete:
        return Titled(
            "Nothing deleted",
            P("No chunk IDs were found."),
            A("Return to management", href="/manage"),
        )

    # Delete only the IDs belonging to this exact hash.
    document_store.delete_documents(ids_to_delete)

    # Verify that no chunks with this hash remain.
    remaining = get_chunks_for_file(file_hash)

    if remaining:
        return Titled(
            "Deletion needs checking",
            P(
                "Some matching chunks remain. Check the terminal "
                "and management page."
            ),
            A("Return to management", href="/manage"),
        )

    return Titled(
        "Document deleted",
        Style(CSS),
        Main(
            Div(
                H1("Deletion complete"),
                P(f"Selected PDF: {expected_name}"),
                P(f"Chunks removed: {len(ids_to_delete)}"),
                P(
                    "Chunks remaining in knowledge base: "
                    f"{document_store.count_documents()}"
                ),
                A("Back to Document Management", href="/manage"),
                cls="success",
            )
        ),
    )


# ==================================================
# START THE APPLICATION
# ==================================================

if __name__ == "__main__":
    # Use port 5001 locally; respect PORT if provided by a host.
    serve(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5001")),
        reload=False,
    )